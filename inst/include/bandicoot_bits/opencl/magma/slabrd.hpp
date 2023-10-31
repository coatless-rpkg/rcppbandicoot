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



// SLABRD reduces the first NB rows and columns of a real general
// m by n matrix A to upper or lower bidiagonal form by an orthogonal
// transformation Q' * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.

inline
magma_int_t
magma_slabrd_gpu
  (
  magma_int_t m, magma_int_t n, magma_int_t nb,
  float     *A, magma_int_t lda,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  float *d, float *e, float *tauq, float *taup,
  float     *X, magma_int_t ldx,
  magmaFloat_ptr dX, size_t dX_offset, magma_int_t lddx,
  float     *Y, magma_int_t ldy,
  magmaFloat_ptr dY, size_t dY_offset, magma_int_t lddy,
  float  *work, magma_int_t /* lwork */,
  magma_queue_t queue
  )
  {
  /* Constants */
  const float c_neg_one = MAGMA_S_NEG_ONE;
  const float c_one     = MAGMA_S_ONE;
  const float c_zero    = MAGMA_S_ZERO;
  const magma_int_t ione = 1;

  /* Local variables */
  magma_int_t i, i1, m_i, m_i1, n_i, n_i1;
  float alpha;

  /* Quick return if possible */
  magma_int_t info = 0;
  if (m <= 0 || n <= 0)
    {
    return info;
    }

  if (m >= n)
    {
    /* Reduce to upper bidiagonal form */
    for (i = 0; i < nb; ++i)
      {
      /* Update A(i:m, i) */
      i1   = i + 1;
      m_i  = m - i;
      m_i1 = m - (i+1);
      n_i1 = n - (i+1);

      blas::gemv('N', m_i, i, c_neg_one,
                 &A[i + (0) * lda], lda,
                 &Y[i + (0) * ldy], ldy, c_one,
                 &A[i + (i) * lda], ione);

      blas::gemv('N', m_i, i, c_neg_one,
                 &X[i + (0) * ldx], ldx,
                 &A[0 + (i) * lda], ione, c_one,
                 &A[i + (i) * lda], ione);

      /* Generate reflection Q(i) to annihilate A(i+1:m, i) */
      alpha = A[i + (i) * lda];
      lapack::larfg(m_i, &alpha, &A[std::min(i+1, m-1) + (i) * lda], ione, &tauq[i]);
      d[i] = alpha;
      if (i+1 < n)
        {
        A[i + (i) * lda] = c_one;

        /* Compute Y(i+1:n, i) */
        // 1. Send the block reflector  A(i+1:m, i) to the GPU ------
        magma_ssetvector( m_i,
                          &A[i + (i) * lda], 1,
                          dA, dA_offset + i + (i) * ldda, 1, queue );
        // 2. Multiply ---------------------------------------------
        magma_sgemv( MagmaConjTrans, m_i, n_i1, c_one,
                     dA, dA_offset + i + (i+1) * ldda,   ldda,
                     dA, dA_offset + i + (i) * ldda, magma_int_t(ione), c_zero,
                     dY, dY_offset + i+1 + (i) * lddy,   magma_int_t(ione), queue );

        // 3. Get the result back ----------------------------------
        magma_sgetmatrix_async( n_i1, 1,
                                dY, dY_offset + i+1 + (i) * lddy, lddy,
                                &Y[i+1 + (i) * ldy],  ldy, queue );

        blas::gemv('C', m_i, i, c_one,
                   &A[i + (0) * lda], lda,
                   &A[i + (i) * lda], ione, c_zero,
                   &Y[0 + (i) * ldy], ione);

        blas::gemv('N', n_i1, i, c_neg_one,
                   &Y[i+1 + (0) * ldy], ldy,
                   &Y[0 + (i) * ldy],   ione, c_zero,
                   work,                ione);
        blas::gemv('C', m_i, i, c_one,
                   &X[i + (0) * ldx], ldx,
                   &A[i + (i) * lda], ione, c_zero,
                   &Y[0 + (i) * ldy], ione);

        // 4. Sync to make sure the result is back ----------------
        magma_queue_sync( queue );

        if (i != 0)
          {
          blas::axpy(n_i1, c_one, work, ione, &Y[i+1 + (i) * ldy], ione);
          }

        blas::gemv('C', i, n_i1, c_neg_one,
                   &A[0 + (i+1) * lda], lda,
                   &Y[0 + (i) * ldy],   ione, c_one,
                   &Y[i+1 + (i) * ldy], ione);
        blas::scal(n_i1, tauq[i], &Y[i+1 + (i) * ldy], ione);

        /* Update A[i + (i+1:n) * lda] */
        blas::gemv('N', n_i1, i1, c_neg_one,
                   &Y[i+1 + (0) * ldy], ldy,
                   &A[i + (0) * lda],   lda, c_one,
                   &A[i + (i+1) * lda], lda);

        blas::gemv('C', i, n_i1, c_neg_one,
                   &A[0 + (i+1) * lda], lda,
                   &X[i + (0) * ldx],   ldx, c_one,
                   &A[i + (i+1) * lda], lda);

        /* Generate reflection P(i) to annihilate A[i + (i+2:n) * lda] */
        alpha = A[i + (i+1) * lda];
        lapack::larfg(n_i1, &alpha, &A[i + (std::min(i+2,n-1)) * lda], lda, &taup[i]);
        e[i] = alpha;
        A[i + (i+1) * lda] = c_one;

        /* Compute X(i+1:m, i) */
        // 1. Send the block reflector  A(i+1:m, i) to the GPU ------
        magma_ssetvector( n_i1,
                          &A[i + (i+1) * lda], lda,
                          dA, dA_offset + i + (i+1) * ldda, ldda, queue );

        // 2. Multiply ---------------------------------------------
        magma_sgemv( MagmaNoTrans, m_i1, n_i1, c_one,
                     dA, dA_offset + i+1 + (i+1) * ldda, ldda,
                     dA, dA_offset + i + (i+1) * ldda, ldda,
                     //dY, 0 + (0) * lddy, 1,
                     c_zero,
                     dX, dX_offset + i+1 + (i) * lddx, ione, queue );

        // 3. Get the result back ----------------------------------
        magma_sgetmatrix_async( m_i1, 1,
                                dX, dX_offset + i+1 + (i) * lddx, lddx,
                                &X[i+1 + (i) * ldx],  ldx, queue );

        blas::gemv('C', n_i1, i1, c_one,
                   &Y[i+1 + (0) * ldy], ldy,
                   &A[i + (i+1) * lda], lda, c_zero,
                   &X[0 + (i) * ldx],   ione);

        blas::gemv('N', m_i1, i1, c_neg_one,
                   &A[i+1 + (0) * lda], lda,
                   &X[0 + (i) * ldx],   ione, c_zero,
                   work,                ione);
        blas::gemv('N', i, n_i1, c_one,
                   &A[0 + (i+1) * lda], lda,
                   &A[i + (i+1) * lda], lda, c_zero,
                   &X[0 + (i) * ldx],   ione);

        // 4. Sync to make sure the result is back ----------------
        magma_queue_sync( queue );
        if ((i+1) != 0)
          {
          blas::axpy(m_i1, c_one, work, ione, &X[i+1 + (i) * ldx], ione);
          }

        blas::gemv('N', m_i1, i, c_neg_one,
                   &X[i+1 + (0) * ldx], ldx,
                   &X[0 + (i) * ldx],   ione, c_one,
                   &X[i+1 + (i) * ldx], ione);
        blas::scal(m_i1, taup[i], &X[i+1 + (i) * ldx], ione);
        }
      }
    }
  else
    {
    /* Reduce to lower bidiagonal form */
    for (i=0; i < nb; ++i)
      {
      /* Update A(i, i:n) */
      i1   = i + 1;
      m_i1 = m - (i+1);
      n_i  = n - i;
      n_i1 = n - (i+1);
      blas::gemv('N', n_i, i, c_neg_one,
                 &Y[i + (0) * ldy], ldy,
                 &A[i + (0) * lda], lda, c_one,
                 &A[i + (i) * lda], lda);
      blas::gemv('C', i, n_i, c_neg_one,
                 &A[0 + (i) * lda], lda,
                 &X[i + (0) * ldx], ldx, c_one,
                 &A[i + (i) * lda], lda);

      /* Generate reflection P(i) to annihilate A(i, i+1:n) */
      alpha = A[i + (i) * lda];
      lapack::larfg(n_i, &alpha, &A[i + (std::min(i+1,n-1)) * lda], lda, &taup[i]);
      d[i] = alpha;
      if (i+1 < m)
        {
        A[i + (i) * lda] = c_one;

        /* Compute X(i+1:m, i) */
        // 1. Send the block reflector  A(i, i+1:n) to the GPU ------
        magma_ssetvector( n_i,
                          &A[i + (i) * lda], lda,
                          dA, dA_offset + i + (i) * ldda, ldda, queue );

        // 2. Multiply ---------------------------------------------
        magma_sgemv( MagmaNoTrans, m_i1, n_i, c_one,
                     dA, dA_offset + i+1 + (i) * ldda, ldda,
                     dA, dA_offset + i + (i) * ldda, ldda,
                     //dY, 0 + (0) * lddy, 1,
                     c_zero,
                     dX, dX_offset + i+1 + (i) * lddx, ione, queue );

        // 3. Get the result back ----------------------------------
        magma_sgetmatrix_async( m_i1, 1,
                                dX, dX_offset + i+1 + (i) * lddx, lddx,
                                &X[i+1 + (i) * ldx],  ldx, queue );

        blas::gemv('C', n_i, i, c_one,
                   &Y[i + (0) * ldy], ldy,
                   &A[i + (i) * lda], lda, c_zero,
                   &X[0 + (i) * ldx], ione);

        blas::gemv('N', m_i1, i, c_neg_one,
                   &A[i+1 + (0) * lda], lda,
                   &X[0 + (i) * ldx],   ione, c_zero,
                   work,                ione);

        blas::gemv('N', i, n_i, c_one,
                   &A[0 + (i) * lda], lda,
                   &A[i + (i) * lda], lda, c_zero,
                   &X[0 + (i) * ldx], ione);

        // 4. Sync to make sure the result is back ----------------
        magma_queue_sync( queue );
        if (i != 0)
          {
          blas::axpy(m_i1, c_one, work, ione, &X[i+1 + (i) * ldx], ione);
          }

        blas::gemv('N', m_i1, i, c_neg_one,
                   &X[i+1 + (0) * ldx], ldx,
                   &X[0 + (i) * ldx],   ione, c_one,
                   &X[i+1 + (i) * ldx], ione);
        blas::scal(m_i1, taup[i], &X[i+1 + (i) * ldx], ione);

        /* Update A[i+1:m + (i) * lda] */
        blas::gemv('N', m_i1, i, c_neg_one,
                   &A[i+1 + (0) * lda], lda,
                   &Y[i + (0) * ldy],   ldy, c_one,
                   &A[i+1 + (i) * lda], ione);
        blas::gemv('N', m_i1, i1, c_neg_one,
                   &X[i+1 + (0) * ldx], ldx,
                   &A[0 + (i) * lda],   ione, c_one,
                   &A[i+1 + (i) * lda], ione);

        /* Generate reflection Q(i) to annihilate A[i+2:m + (i) * lda] */
        alpha = A[i+1 + (i) * lda];
        lapack::larfg(m_i1, &alpha, &A[std::min(i+2, m-1) + (i) * lda], ione, &tauq[i]);
        e[i] = alpha;
        A[i+1 + (i) * lda] = c_one;

        /* Compute Y(i+1:n, i) */
        // 1. Send the block reflector  A(i+1:m, i) to the GPU ------
        magma_ssetvector( m_i1,
                          &A[i+1 + (i) * lda], 1,
                          dA, dA_offset + i+1 + (i) * ldda, 1, queue );

        // 2. Multiply ---------------------------------------------
        magma_sgemv( MagmaConjTrans, m_i1, n_i1, c_one,
                     dA, dA_offset + i+1 + (i+1) * ldda, ldda,
                     dA, dA_offset + i+1 + (i) * ldda, ione, c_zero,
                     dY, dY_offset + i+1 + (i) * lddy, ione, queue );

        // 3. Get the result back ----------------------------------
        magma_sgetmatrix_async( n_i1, 1,
                                dY, dY_offset + i+1 + (i) * lddy, lddy,
                                &Y[i+1 + (i) * ldy],  ldy, queue );

        blas::gemv('C', m_i1, i, c_one,
                   &A[i+1 + (0) * lda], lda,
                   &A[i+1 + (i) * lda], ione, c_zero,
                   &Y[0 + (i) * ldy],   ione);
        blas::gemv('N', n_i1, i, c_neg_one,
                   &Y[i+1 + (0) * ldy], ldy,
                   &Y[0 + (i) * ldy],   ione, c_zero,
                   work,                ione);

        blas::gemv('C', m_i1, i1, c_one,
                   &X[i+1 + (0) * ldx], ldx,
                   &A[i+1 + (i) * lda], ione, c_zero,
                   &Y[0 + (i) * ldy],   ione);

        // 4. Sync to make sure the result is back ----------------
        magma_queue_sync( queue );
        if (i != 0)
          {
          blas::axpy(n_i1, c_one, work, ione, &Y[i+1 + (i) * ldy], ione);
          }

        blas::gemv('C', i1, n_i1, c_neg_one,
                   &A[0 + (i+1) * lda], lda,
                   &Y[0 + (i) * ldy],   ione, c_one,
                   &Y[i+1 + (i) * ldy], ione);
        blas::scal(n_i1, tauq[i], &Y[i+1 + (i) * ldy], ione);
      }
    }
  }

  return info;
  } /* magma_slabrd_gpu */
