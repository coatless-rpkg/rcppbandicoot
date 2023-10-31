// Copyright 2023 Ryan Curtin (http://www.ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// This file contains source code adapted from
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.7 (2022-11-09).
// clMAGMA 1.3 and MAGMA 2.7 are distributed under a
// 3-clause BSD license as follows:
//
//  -- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2015
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.



inline
magma_int_t
magma_get_sgetrf_nb(magma_int_t m)
  {
  if      (m <= 3200) return 128;
  else if (m <  9000) return 256;
  else                return 320;
  }



// Purpose
// -------
// SGETRF computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges.
//
// The factorization has the form
//     A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 3 BLAS version of the algorithm.

inline
magma_int_t
magma_sgetrf_gpu
  (
  magma_int_t m, magma_int_t n,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  magma_int_t *ipiv,
  magma_int_t *info
  )
  {
  float c_one     = MAGMA_S_ONE;
  float c_neg_one = MAGMA_S_NEG_ONE;

  magma_int_t iinfo;
  magma_int_t maxm, maxn, minmn;
  magma_int_t i, j, jb, rows, lddat, ldwork;
  magmaFloat_ptr dAT=NULL, dAP=NULL;
  size_t dAT_offset;
  float *work=NULL; // hybrid

  /* Check arguments */
  *info = 0;
  if (m < 0)
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (ldda < std::max(1, m))
    {
    *info = -4;
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  /* Quick return if possible */
  if (m == 0 || n == 0)
    {
    return *info;
    }

  /* Function Body */
  minmn = std::min( m, n );

  magma_int_t nb = magma_get_sgetrf_nb( m );

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();

  if (nb <= 1 || nb >= std::min(m,n) )
    {
    if ( MAGMA_SUCCESS != magma_smalloc_cpu( &work, m*n ))
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      magma_queue_destroy( queues[0] );
      return *info;
      }
    magma_sgetmatrix( m, n, dA, dA_offset, ldda, work, m, queues[0] );
    lapack::getrf(m, n, work, m, ipiv, info);
    magma_ssetmatrix( m, n, work, m, dA, dA_offset, ldda, queues[0] );
    magma_free_cpu( work );  work=NULL;

    magma_queue_destroy( queues[0] );
    return *info;
    }
  else
    {
    queues[1] = magma_queue_create();

    /* Use blocked code. */
    maxm = magma_roundup( m, 32 );
    maxn = magma_roundup( n, 32 );

    if (MAGMA_SUCCESS != magma_smalloc( &dAP, nb*maxm ))
      {
      *info = MAGMA_ERR_DEVICE_ALLOC;
      goto cleanup;
      }

    // square matrices can be done in place;
    // rectangular requires copy to transpose
    if ( m == n )
      {
      dAT = dA;
      dAT_offset = dA_offset;
      lddat = ldda;
      magmablas_stranspose_inplace( m, dAT, dAT_offset, lddat, queues[0] );
      }
    else
      {
      lddat = maxn;  // N-by-M
      dAT_offset = 0;
      if (MAGMA_SUCCESS != magma_smalloc( &dAT, lddat*maxm ))
        {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        goto cleanup;
        }
      magmablas_stranspose( m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat, queues[0] );
      }
    magma_queue_sync( queues[0] );  // finish transpose

    ldwork = maxm;
    if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, ldwork*nb ))
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      goto cleanup;
      }

    for( j=0; j < minmn-nb; j += nb )
      {
      // get j-th panel from device
      magmablas_stranspose( nb, m-j, dAT, dAT_offset + j * lddat + j, lddat, dAP, 0, maxm, queues[1] );
      magma_queue_sync( queues[1] );  // wait for transpose
      magma_sgetmatrix_async( m-j, nb, dAP, 0, maxm, work, ldwork, queues[0] );

      if ( j > 0 )
        {
        magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                     n-(j+nb), nb,
                     c_one, dAT, dAT_offset + (j-nb) * lddat + (j-nb), lddat,
                            dAT, dAT_offset + (j-nb) * lddat + (j+nb), lddat, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     n-(j+nb), m-j, nb,
                     c_neg_one, dAT, dAT_offset + (j-nb) * lddat + (j+nb), lddat,
                                dAT, dAT_offset + (j   ) * lddat + (j-nb), lddat,
                     c_one,     dAT, dAT_offset + (j   ) * lddat + (j+nb), lddat, queues[1] );
        }

      rows = m - j;

      // do the cpu part
      magma_queue_sync( queues[0] );  // wait to get work
      lapack::getrf(rows, nb, work, ldwork, ipiv + j, &iinfo);
      if ( *info == 0 && iinfo > 0 )
        {
        *info = iinfo + j;
        }

      // send j-th panel to device
      magma_ssetmatrix_async( m-j, nb, work, ldwork, dAP, 0, maxm, queues[0] );

      for( i=j; i < j + nb; ++i )
        {
        ipiv[i] += j;
        }
      magmablas_slaswp( n, dAT, dAT_offset, lddat, j + 1, j + nb, ipiv, 1, queues[1] );

      magma_queue_sync( queues[0] );  // wait to set dAP
      magmablas_stranspose( m-j, nb, dAP, 0, maxm, dAT, dAT_offset + j * lddat + j, lddat, queues[1] );

      // do the small non-parallel computations (next panel update)
      if ( j + nb < minmn - nb )
        {
        magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                     nb, nb,
                     c_one, dAT, dAT_offset + j * lddat + (j   ), lddat,
                            dAT, dAT_offset + j * lddat + (j+nb), lddat, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     nb, m-(j+nb), nb,
                     c_neg_one, dAT, dAT_offset + (j   ) * lddat + (j+nb), lddat,
                                dAT, dAT_offset + (j+nb) * lddat + (j   ), lddat,
                     c_one,     dAT, dAT_offset + (j+nb) * lddat + (j+nb), lddat, queues[1] );
        }
      else
        {
        magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                     n-(j+nb), nb,
                     c_one, dAT, dAT_offset + j * lddat + (j   ), lddat,
                            dAT, dAT_offset + j * lddat + (j+nb), lddat, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     n-(j+nb), m-(j+nb), nb,
                     c_neg_one, dAT, dAT_offset + (j   ) * lddat + (j+nb), lddat,
                                dAT, dAT_offset + (j+nb) * lddat + (j   ), lddat,
                     c_one,     dAT, dAT_offset + (j+nb) * lddat + (j+nb), lddat, queues[1] );
        }
      }

    jb = std::min( m-j, n-j );
    if ( jb > 0 )
      {
      rows = m - j;

      magmablas_stranspose( jb, rows, dAT, dAT_offset + j * lddat + j, lddat, dAP, 0, maxm, queues[1] );
      magma_sgetmatrix( rows, jb, dAP, 0, maxm, work, ldwork, queues[1] );

      // do the cpu part
      lapack::getrf(rows, jb, work, ldwork, ipiv + j, &iinfo);
      if ( *info == 0 && iinfo > 0 )
          *info = iinfo + j;

      for( i=j; i < j + jb; ++i )
        {
        ipiv[i] += j;
        }
      magmablas_slaswp( n, dAT, dAT_offset, lddat, j + 1, j + jb, ipiv, 1, queues[1] );

      // send j-th panel to device
      magma_ssetmatrix( rows, jb, work, ldwork, dAP, 0, maxm, queues[1] );

      magmablas_stranspose( rows, jb, dAP, 0, maxm, dAT, dAT_offset + j * lddat + j, lddat, queues[1] );

      magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                   n-j-jb, jb,
                   c_one, dAT, dAT_offset + j * lddat + (j),    lddat,
                          dAT, dAT_offset + j * lddat + (j+jb), lddat, queues[1] );
    }

    // undo transpose
    if ( m == n )
      {
      magmablas_stranspose_inplace( m, dAT, dAT_offset, lddat, queues[1] );
      }
    else
      {
      magmablas_stranspose( n, m, dAT, dAT_offset, lddat, dA, dA_offset, ldda, queues[1] );
      }
    }

cleanup:
  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  magma_free( dAP );
  if (m != n)
    {
    magma_free( dAT );
    }

  magma_free_pinned( work );

  return *info;
  } /* magma_sgetrf_gpu */
