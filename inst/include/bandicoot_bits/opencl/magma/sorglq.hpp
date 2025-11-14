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



// SORGLQ generates an M-by-N real matrix Q with orthonormal rows,
// which is defined as the first M rows of a product of K elementary
// reflectors of order N
//
//        Q  =  H(k)**H . . . H(2)**H H(1)**H
//
// as returned by SGELQF.



inline
magma_int_t
magma_get_sgelqf_nb(magma_int_t m, magma_int_t n)
  {
  return magma_get_sgeqrf_nb(m, n);
  }



inline
magma_int_t
magma_sorglq
  (
  magma_int_t m, magma_int_t n, magma_int_t k,
  float *A, magma_int_t lda,
  float *tau,
  float *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  // Constants
  const float c_zero = MAGMA_S_ZERO;
  const float c_one  = MAGMA_S_ONE;

  // Local variables
  bool lquery;
  magma_int_t i, ib, ki, ldda, lddwork, lwkopt, mib, nb, n_i;
  magmaFloat_ptr dA = NULL;
  float* work_local = NULL;
  magma_queue_t queue = NULL;

  // Test the input arguments
  *info = 0;
  nb = magma_get_sgelqf_nb(m, n);
  lwkopt = m*nb;
  work[0] = magma_smake_lwork( lwkopt );
  lquery = (lwork == -1);
  if (m < 0)
    *info = -1;
  else if (n < 0 || n < m)
    *info = -2;
  else if (k < 0 || k > m)
    *info = -3;
  else if (lda < std::max( 1, m ))
    *info = -5;
  else if (lwork < std::max( 1, lwkopt ) && !lquery)
    *info = -8;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  // Quick return if possible
  if (m <= 0)
    {
    work[0] = c_one;
    return *info;
    }

  // Need at least nb*nb to store T.
  // For better LAPACK compatibility, which needs M*NB,
  // allow lwork < NB*NB and allocate here if needed.
  if (lwork < nb*nb)
    {
    if (MAGMA_SUCCESS != magma_smalloc_cpu( &work_local, lwkopt ))
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      goto cleanup;
      }
    work = work_local;
    }

  // Allocate GPU work space
  // ldda*n     for matrix dA
  // nb*n       for dV
  // lddwork*nb for dW larfb workspace
  ldda    = magma_roundup( m, 32 );
  lddwork = magma_roundup( m, 32 );
  if (MAGMA_SUCCESS != magma_smalloc( &dA, ldda*n + n*nb + lddwork*nb + nb*nb ))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    goto cleanup;
    }

  magmaFloat_ptr dV, dW, dT;
  size_t dV_offset, dW_offset, dT_offset;
  dV = dA;
  dV_offset = ldda*n;
  dW = dA;
  dW_offset = ldda*n + n*nb;
  dT = dA;
  dT_offset = ldda*n + n*nb + lddwork*nb;

  queue = magma_queue_create();

  magmablas_slaset( MagmaFull, m, n, Datum<double>::nan, Datum<double>::nan, dA, 0, ldda, queue );

  // all columns are handled by blocked method.
  // ki is start of last (partial) block
  ki = ((k - 1) / nb) * nb;

  // Use blocked code
  for( i = ki; i >= 0; i -= nb )
    {
    ib = std::min( nb, k-i );
    // first block has extra rows to update
    mib = ib;
    if ( i == ki )
      {
      mib = m - i;
      }

    // Send current panel of V (block row) to the GPU
    lapack::laset('L', ib, ib, c_zero, c_one, &A[i + i * lda], lda);
    // TODO: having this _async was causing numerical errors. Why?
    magma_ssetmatrix( ib, n-i,
                      &A[i + i * lda], lda,
                      dV, dV_offset,  nb, queue );

    // Form the triangular factor of the block reflector
    // H = H(i) H(i+1) . . . H(i+ib-1)
    n_i = n - i;
    lapack::larft('F', 'R', n_i, ib,
                  &A[i + i * lda], lda, &tau[i], work, nb);
    magma_ssetmatrix_async( ib, ib,
                            work, nb,
                            dT, dT_offset,   nb, queue );

    // set panel of A (block row) to identity
    magmablas_slaset( MagmaFull, mib, i,   c_zero, c_zero, dA, i,            ldda, queue );
    magmablas_slaset( MagmaFull, mib, n-i, c_zero, c_one,  dA, i + i * ldda, ldda, queue );

    if (i < m)
      {
      // Apply H**H to A(i:m,i:n) from the right
      magma_slarfb_gpu( MagmaRight, MagmaConjTrans, MagmaForward, MagmaRowwise,
                        m-i, n-i, ib,
                        dV, dV_offset, nb,
                        dT, dT_offset, nb,
                        dA, i + i * ldda, ldda,
                        dW, dW_offset, lddwork, queue );
      }
    }

  // copy result back to CPU
  magma_sgetmatrix( m, n, dA, 0, ldda, A, lda, queue );

cleanup:
  magma_queue_sync( queue );
  magma_queue_destroy( queue );

  work[0] = magma_smake_lwork( lwkopt );  // before free( work_local )

  magma_free( dA );
  magma_free_cpu( work_local );

  return *info;
  }
