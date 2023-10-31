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



// DGELQF computes an LQ factorization of a DOUBLE PRECISION M-by-N matrix dA:
// dA = L * Q.



inline
magma_int_t
magma_dgelqf
  (
  magma_int_t m, magma_int_t n,
  double *A,    magma_int_t lda,   double *tau,
  double *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one = MAGMA_D_ONE;

  /* Local variables */
  magmaDouble_ptr dA=NULL, dAT=NULL;
  size_t dAT_offset;
  magma_int_t min_mn, maxm, maxn, maxdim, nb;
  magma_int_t iinfo, ldda, lddat;

  /* Function Body */
  *info = 0;
  nb = magma_get_dgelqf_nb( m, n );
  min_mn = std::min( m, n );

  work[0] = magma_dmake_lwork( m*nb );
  bool lquery = (lwork == -1);
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (lda < std::max(1,m))
    *info = -4;
  else if (lwork < std::max(1,m) && ! lquery)
    *info = -7;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  /* Quick return if possible */
  if (min_mn == 0)
    {
    work[0] = c_one;
    return *info;
    }

  maxm = magma_roundup( m, 32 );
  maxn = magma_roundup( n, 32 );
  maxdim = std::max( maxm, maxn );

  magma_queue_t queue = magma_queue_create();

  // copy to GPU and transpose
  if (maxdim*maxdim < 2*maxm*maxn)
    {
    // close to square, do everything in-place
    ldda  = maxdim;
    lddat = maxdim;

    if (MAGMA_SUCCESS != magma_dmalloc( &dA, maxdim*maxdim ))
      {
      *info = MAGMA_ERR_DEVICE_ALLOC;
      goto cleanup;
      }

    magma_dsetmatrix( m, n, A, lda, dA, 0, ldda, queue );
    dAT = dA;
    dAT_offset = 0;
    magmablas_dtranspose_inplace( lddat, dAT, dAT_offset, lddat, queue );
    }
  else
    {
    // rectangular, do everything out-of-place
    ldda  = maxm;
    lddat = maxn;

    if (MAGMA_SUCCESS != magma_dmalloc( &dA, 2*maxn*maxm ))
      {
      *info = MAGMA_ERR_DEVICE_ALLOC;
      goto cleanup;
      }

    magma_dsetmatrix( m, n, A, lda, dA, 0, ldda, queue );

    dAT = dA;
    dAT_offset = maxn * maxm;
    magmablas_dtranspose( m, n, dA, 0, ldda, dAT, dAT_offset, lddat, queue );
    }

    // factor QR
    magma_queue_sync(queue);
    magma_dgeqrf2_gpu( n, m, dAT, dAT_offset, lddat, tau, &iinfo );
    assert( iinfo >= 0 );
    if ( iinfo > 0 )
      {
      *info = iinfo;
      }

    // undo transpose
    if (maxdim*maxdim < 2*maxm*maxn)
      {
      magmablas_dtranspose_inplace( lddat, dAT, dAT_offset, lddat, queue );
      magma_dgetmatrix( m, n, dA, 0, ldda, A, lda, queue );
      }
    else
      {
      magmablas_dtranspose( n, m, dAT, dAT_offset, lddat, dA, 0, ldda, queue );
      magma_dgetmatrix( m, n, dA, 0, ldda, A, lda, queue );
      }

cleanup:
  magma_queue_destroy( queue );
  magma_free( dA );

  return *info;
  }



inline
magma_int_t
magma_dgelqf_gpu
  (
  magma_int_t m,
  magma_int_t n,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  double *tau,
  double *work,
  magma_int_t lwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one = MAGMA_D_ONE;

  /* Local variables */
  magmaDouble_ptr dAT = NULL;
  size_t dAT_offset;
  magma_int_t min_mn, maxm, maxn, nb;
  magma_int_t iinfo;

  *info = 0;
  nb = magma_get_dgelqf_nb(m, n);
  min_mn = std::min(m, n);

  work[0] = magma_dmake_lwork(m * nb);
  bool lquery = (lwork == -1);
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (ldda < std::max(1, m))
    *info = -4;
  else if (lwork < std::max(1, m) && !lquery)
    *info = -7;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  /*  Quick return if possible */
  if (min_mn == 0)
    {
    work[0] = c_one;
    return *info;
    }

  maxm = magma_roundup(m, 32);
  maxn = magma_roundup(n, 32);

  magma_int_t lddat = maxn;

  magma_queue_t queue = magma_queue_create();

  if (m == n)
    {
    dAT = dA;
    dAT_offset = dA_offset;
    lddat = ldda;
    magmablas_dtranspose_inplace(m, dAT, dAT_offset, ldda, queue);
    }
  else
    {
    if (MAGMA_SUCCESS != magma_dmalloc(&dAT, maxm*maxn))
      {
      *info = MAGMA_ERR_DEVICE_ALLOC;
      goto cleanup;
      }
    dAT_offset = 0;

    magmablas_dtranspose(m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat, queue);
    }

  magma_queue_sync(queue);
  magma_dgeqrf2_gpu(n, m, dAT, dAT_offset, lddat, tau, &iinfo);
  assert(iinfo >= 0);
  if (iinfo > 0)
    {
    *info = iinfo;
    }

  if (m == n)
    {
    magmablas_dtranspose_inplace(m, dAT, dAT_offset, lddat, queue);
    }
  else
    {
    magmablas_dtranspose(n, m, dAT, dAT_offset, lddat, dA, dA_offset, ldda, queue);
    magma_free(dAT);
    }

cleanup:
  magma_queue_destroy( queue );

  return *info;
} /* magma_dgelqf_gpu */
