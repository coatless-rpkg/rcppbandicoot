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



// SORGQR generates an M-by-N REAL matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by SGEQRF.
//
// This version recomputes the T matrices on the CPU and sends them to the GPU.

inline
magma_int_t
magma_sorgqr2
  (
  magma_int_t m, magma_int_t n, magma_int_t k,
  float *A, magma_int_t lda,
  float *tau,
  magma_int_t *info
  )
  {
  float c_zero = MAGMA_S_ZERO;
  float c_one  = MAGMA_S_ONE;

  magma_int_t nb = magma_get_sgeqrf_nb( m, n );

  magma_int_t  m_kk, n_kk, k_kk, mi;
  magma_int_t lwork, ldda;
  magma_int_t i, ib, ki, kk;  //, iinfo;
  magma_int_t lddwork;
  magmaFloat_ptr dA, dV, dW, dT;
  float* T;
  size_t dV_offset, dW_offset, dT_offset;
  float *work;

  *info = 0;
  if (m < 0)
    *info = -1;
  else if ((n < 0) || (n > m))
    *info = -2;
  else if ((k < 0) || (k > n))
    *info = -3;
  else if (lda < std::max(1,m))
    *info = -5;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  if (n <= 0)
    {
    return *info;
    }

  // first kk columns are handled by blocked method.
  // ki is start of 2nd-to-last block
  if ((nb > 1) && (nb < k))
    {
    ki = (k - nb - 1) / nb * nb;
    kk = std::min(k, ki + nb);
    }
  else
    {
    ki = 0;
    kk = 0;
    }

  // Allocate GPU work space
  // ldda*n     for matrix dA
  // ldda*nb    for dV
  // lddwork*nb for dW larfb workspace
  ldda    = magma_roundup( m, 32 );
  lddwork = magma_roundup( n, 32 );
  if (MAGMA_SUCCESS != magma_smalloc( &dA, ldda*n + ldda*nb + lddwork*nb + nb*nb))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }
  dV = dA;
  dV_offset = ldda*n;
  dW = dA;
  dW_offset = ldda*n + ldda*nb;
  dT = dA;
  dT_offset = ldda*n + ldda*nb + lddwork*nb;

  // Allocate CPU work space
  lwork = (n+m+nb) * nb;
  magma_smalloc_cpu( &work, lwork );

  T = work;

  if (work == NULL)
    {
    magma_free( dA );
    magma_free_cpu( work );
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }
  float *V = work + (n+nb)*nb;

  magma_queue_t queue = magma_queue_create();

  // Use unblocked code for the last or only block.
  if (kk < n)
    {
    m_kk = m - kk;
    n_kk = n - kk;
    k_kk = k - kk;
    /*
        lapackf77_dorgqr( &m_kk, &n_kk, &k_kk,
                          A(kk, kk), &lda,
                          &tau[kk], work, &lwork, &iinfo );
    */
    lapack::lacpy('A', m_kk, k_kk, &A[kk + kk * lda], lda, V, m_kk);
    lapack::laset('A', m_kk, n_kk, c_zero, c_one, &A[kk + kk * lda], lda);

    lapack::larft('F', 'C',
                  m_kk, k_kk,
                  V, m_kk, &tau[kk], work, k_kk);
    lapack::larfb('L', 'N', 'F', 'C',
                  m_kk, n_kk, k_kk,
                  V, m_kk, work, k_kk, &A[kk + kk * lda], lda, work + k_kk * k_kk, n_kk);

    if (kk > 0)
      {
      magma_ssetmatrix( m_kk, n_kk,
                        &A[kk + kk * lda],  lda,
                        dA, kk + kk * ldda, ldda, queue );

      // Set A(1:kk,kk+1:n) to zero.
      magmablas_slaset( MagmaFull, kk, n - kk, c_zero, c_zero, dA, 0 + kk * ldda, ldda, queue );
    }
  }

  if (kk > 0)
    {
    // Use blocked code
    // queue: set Aii (V) --> laset --> laset --> larfb --> [next]
    // CPU has no computation

    for (i = ki; i >= 0; i -= nb)
      {
      ib = std::min(nb, k - i);

      // Send current panel to the GPU
      mi = m - i;
      lapack::laset('U', ib, ib, c_zero, c_one, &A[i + i * lda], lda);
      magma_ssetmatrix_async( mi, ib,
                              &A[i + i * lda], lda,
                              dV, dV_offset,   ldda, queue );
      lapack::larft(MagmaForwardStr[0], MagmaColumnwiseStr[0],
                    mi, ib,
                    &A[i + i * lda], lda, &tau[i], T, nb);
      magma_ssetmatrix_async( ib, ib,
                              T, nb,
                              dT, dT_offset, nb, queue );

      // set panel to identity
      magmablas_slaset( MagmaFull, i,  ib, c_zero, c_zero, dA, 0 + i * ldda, ldda, queue );
      magmablas_slaset( MagmaFull, mi, ib, c_zero, c_one,  dA, i + i * ldda, ldda, queue );

      magma_queue_sync( queue );
      if (i < n) {
          // Apply H to A(i:m,i:n) from the left
          magma_slarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                            mi, n-i, ib,
                            dV, dV_offset,    ldda, dT, dT_offset, nb,
                            dA, i + i * ldda, ldda, dW, dW_offset, lddwork, queue );
      }
    }

    // copy result back to CPU
    magma_sgetmatrix( m, n, dA, 0, ldda, A, lda, queue );
  }

  magma_queue_destroy( queue );
  magma_free( dA );
  magma_free_cpu( work );

  return *info;
} /* magma_sorgqr */
