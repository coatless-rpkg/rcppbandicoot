// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.2 (2016-11-20).
// clMAGMA 1.3 and MAGMA 2.2 are distributed under a
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
magma_get_dpotrf_nb( magma_int_t m )
  {
  if      (m <= 4256) return 128;
  else                return 256;
  }



inline
magma_int_t
magma_dpotrf_gpu
  (
  magma_uplo_t uplo,
  magma_int_t n,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one     = MAGMA_D_ONE;
  const double c_neg_one = MAGMA_D_NEG_ONE;
  const double d_one     =  1.0;
  const double d_neg_one = -1.0;

  /* Local variables */
  const char* uplo_ = lapack_uplo_const( uplo );
  bool upper = (uplo == MagmaUpper);

  magma_int_t j, jb, nb;
  double *work;

  *info = 0;
  if (!upper && uplo != MagmaLower)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (ldda < std::max(1, n))
    *info = -4;

  if (*info != 0)
    {
    // magma_xerbla( __func__, -(*info) );
    return *info;
    }

  nb = magma_get_dpotrf_nb( n );

  if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb ))
    {
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if (nb <= 1 || nb >= n)
    {
    /* Use unblocked code. */
    magma_dgetmatrix( n, n, dA, dA_offset, ldda, work, n, queues[0] );
    lapack::potrf(uplo_[0], n, work, n, info);
    magma_dsetmatrix( n, n, work, n, dA, dA_offset, ldda, queues[0] );
    }
  else
    {
    /* Use blocked code. */
    if (upper)
      {
      //=========================================================
      /* Compute the Cholesky factorization A = U'*U. */
      for (j = 0; j < n; j += nb)
        {
        // apply all previous updates to diagonal block,
        // then transfer it to CPU
        jb = std::min( nb, n-j );
        if (j > 0)
          {
          magma_dsyrk( MagmaUpper, MagmaConjTrans, jb, j,
                       d_neg_one, dA, dA_offset + j * ldda,     ldda,
                       d_one,     dA, dA_offset + j + j * ldda, ldda, queues[1] );
          }

        magma_queue_sync( queues[1] );
        magma_dgetmatrix_async( jb, jb,
                                dA, dA_offset + j + j * ldda, ldda,
                                work,                     jb, queues[0] );

        // apply all previous updates to block row right of diagonal block
        if (j+jb < n)
          {
          magma_dgemm( MagmaConjTrans, MagmaNoTrans,
                       jb, n-j-jb, j,
                       c_neg_one, dA, dA_offset + j * ldda,            ldda,
                                  dA, dA_offset + (j + jb) * ldda,     ldda,
                       c_one,     dA, dA_offset + j + (j + jb) * ldda, ldda, queues[1] );
          }

        // simultaneous with above dgemm, transfer diagonal block,
        // factor it on CPU, and test for positive definiteness
        magma_queue_sync( queues[0] );
        lapack::potrf(MagmaUpperStr[0], jb, work, jb, info);

        magma_dsetmatrix_async( jb, jb,
                                work,                     jb,
                                dA, dA_offset + j + j * ldda, ldda, queues[1] );
        if (*info != 0)
          {
          *info = *info + j;
          break;
          }

        // apply diagonal block to block row right of diagonal block
        if (j+jb < n)
          {
          magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                       jb, n-j-jb,
                       c_one, dA, dA_offset + j + j * ldda,        ldda,
                              dA, dA_offset + j + (j + jb) * ldda, ldda, queues[1] );
          }
        }
      }
    else
      {
      //=========================================================
      // Compute the Cholesky factorization A = L*L'.
      for (j = 0; j < n; j += nb)
        {
        // apply all previous updates to diagonal block,
        // then transfer it to CPU
        jb = std::min( nb, n - j );
        if (j > 0)
          {
          magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                       d_neg_one, dA, dA_offset + j,            ldda,
                       d_one,     dA, dA_offset + j + j * ldda, ldda, queues[1] );
          }

        magma_queue_sync( queues[1] );
        magma_dgetmatrix_async( jb, jb,
                                dA, dA_offset + j + j * ldda, ldda,
                                work,                     jb, queues[0] );

        // apply all previous updates to block column below diagonal block
        if (j+jb < n)
          {
          magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                       n-j-jb, jb, j,
                       c_neg_one, dA, dA_offset + j + jb,            ldda,
                                  dA, dA_offset + j,                 ldda,
                       c_one,     dA, dA_offset + j + jb + j * ldda, ldda, queues[1] );
          }

        // simultaneous with above dgemm, transfer diagonal block,
        // factor it on CPU, and test for positive definiteness
        magma_queue_sync( queues[0] );
        lapack::potrf(MagmaLowerStr[0], jb, work, jb, info);
        magma_dsetmatrix_async( jb, jb,
                                work,                     jb,
                                dA, dA_offset + j + j * ldda, ldda, queues[1] );
        if (*info != 0)
          {
          *info = *info + j;
          break;
          }

        // apply diagonal block to block column below diagonal
        if (j + jb < n)
          {
          magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                       n-j-jb, jb,
                       c_one, dA, dA_offset + j + j * ldda,      ldda,
                              dA, dA_offset + j + jb + j * ldda, ldda, queues[1] );
          }
        }
      }
    }

  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  magma_free_pinned( work );

  return *info;
  } /* magma_dpotrf_gpu */
