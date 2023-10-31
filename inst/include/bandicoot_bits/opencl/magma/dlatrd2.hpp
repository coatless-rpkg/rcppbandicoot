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



// Purpose
// -------
// DLATRD2 reduces NB rows and columns of a real symmetric matrix A to
// symmetric tridiagonal form by an orthogonal similarity
// transformation Q' * A * Q, and returns the matrices V and W which are
// needed to apply the transformation to the unreduced part of A.
//
// If UPLO = MagmaUpper, DLATRD reduces the last NB rows and columns of a
// matrix, of which the upper triangle is supplied;
// if UPLO = MagmaLower, DLATRD reduces the first NB rows and columns of a
// matrix, of which the lower triangle is supplied.
//
// This is an auxiliary routine called by DSYTRD2_GPU. It uses an
// accelerated HEMV that needs extra memory.



inline
magma_int_t
magma_dlatrd2
  (
  magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
  double *A,  magma_int_t lda,
  double *e, double *tau,
  double *W,  magma_int_t ldw,
  double *work, magma_int_t lwork,
  magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_ptr dW, size_t dW_offset, magma_int_t lddw,
  magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
  magma_queue_t queue
  )
  {
  /* Constants */
  const double c_neg_one = MAGMA_D_NEG_ONE;
  const double c_one     = MAGMA_D_ONE;
  const double c_zero    = MAGMA_D_ZERO;
  const magma_int_t ione = 1;

  /* Local variables */
  double alpha, value;
  magma_int_t i, i_n, i_1, iw;

  /* Check arguments */
  magma_int_t info = 0;
  if ( uplo != MagmaLower && uplo != MagmaUpper )
    {
    info = -1;
    }
  else if ( n < 0 )
    {
    info = -2;
    }
  else if ( nb < 1 )
    {
    info = -3;
    }
  else if ( lda < std::max(1,n) )
    {
    info = -5;
    }
  else if ( ldw < std::max(1,n) )
    {
    info = -9;
    }
  else if ( lwork < std::max(1,n) )
    {
    info = -11;
    }
  else if ( ldda < std::max(1,n) )
    {
    info = -13;
    }
  else if ( lddw < std::max(1,n) )
    {
    info = -15;
    }
  else if ( ldwork < ldda*magma_ceildiv(n,64) )
    {
    info = -17;
    }

  if (info != 0)
    {
    //magma_xerbla( __func__, -(info) );
    return info;
    }

  /* Quick return if possible */
  if (n == 0)
    {
    return info;
    }

  if (uplo == MagmaUpper)
    {
    /* Reduce last NB columns of upper triangle */
    for (i = n-1; i >= n - nb; --i)
      {
      i_1 = i + 1;
      i_n = n - i - 1;

      iw = i - n + nb;
      if (i < n-1)
        {
        /* Update A(1:i,i) */
        blas::gemv('N', i_1, i_n,
                   c_neg_one, A + (i+1) * lda,        lda,
                              W + (i) + (iw+1) * ldw, ldw, c_one,
                              A + (i) * lda,          ione);
        blas::gemv('N', i_1, i_n,
                   c_neg_one, W + (iw+1) * ldw,       ldw,
                              A + i + (i+1) * lda,    lda, c_one,
                              A + (i) * lda,          ione);
        }

      if (i > 0)
        {
        /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
        alpha = *(A + (i-1) + i * lda);

        lapack::larfg(i, &alpha, A + (i) * lda, ione, &tau[i - 1]);

        e[i-1] = alpha;
        *(A + (i-1) + i * lda) = MAGMA_D_ONE;

        /* Compute W(1:i-1,i) */
        // 1. Send the block reflector  A(0:n-i-1,i) to the GPU
        magma_dsetvector_async( i, A + (i) * lda, 1, dA, dA_offset + (i) * ldda, 1, queue );

        magmablas_dsymv_work( MagmaUpper, i, c_one, dA, dA_offset, ldda,
                              dA, dA_offset + i * ldda, ione, c_zero, dW, dW_offset + iw * lddw, ione,
                              dwork, dwork_offset, ldwork, queue );

        // 2. Start getting the result back (asynchronously)
        magma_dgetmatrix_async( i, 1,
                                dW, dW_offset + iw * lddw, lddw,
                                W + (iw) * ldw,  ldw, queue );

        if (i < n-1)
          {
          blas::gemv('C', i, i_n,
                     c_one, W + (iw+1) * ldw,       ldw,
                            A + i * lda,            ione, c_zero,
                            W + (i+1) * (iw) * ldw, ione);
          }

        // 3. Here we need dsymv result W(0, iw)
        magma_queue_sync( queue );

        if (i < n-1)
          {
          blas::gemv('N', i, i_n,
                     c_neg_one, A + (i+1) * lda,        lda,
                                W + (i+1) + (iw) * ldw, ione, c_one,
                                W + (iw) * ldw,         ione);
          blas::gemv('C', i, i_n,
                     c_one,     A + (i+1) * lda,        lda,
                                A + (i) * lda,          ione, c_zero,
                                W + (i+1) + (iw) * ldw, ione);
          blas::gemv('N', i, i_n,
                     c_neg_one, W + (iw+1) * ldw,       ldw,
                                W + (i+1) + (iw) * ldw, ione, c_one,
                                W + (iw) * ldw,         ione);
          }

        blas::scal(i, tau[i - 1], W + (iw) * ldw, ione);

        value = magma_cblas_ddot( i, W + (iw) * ldw, ione, A + (i) * lda, ione );
        alpha = tau[i - 1] * -0.5f * value;
        blas::axpy(i, alpha, A + (i) * lda,  ione,
                             W + (iw) * ldw, ione);
        }
      }
    }
  else
    {
    /*  Reduce first NB columns of lower triangle */
    for (i = 0; i < nb; ++i)
      {
      /* Update A(i:n,i) */
      i_n = n - i;
      blas::gemv('N', i_n, i,
                 c_neg_one, A + (i),             lda,
                            W + i,               ldw, c_one,
                            A + (i) + (i) * lda, ione);
      blas::gemv('N', i_n, i,
                 c_neg_one, W + i,               ldw,
                            A + i,               lda, c_one,
                            A + (i) + (i) * lda, ione);

      if (i < n-1)
        {
        /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
        i_n = n - i - 1;
        alpha = *(A + (i+1) + i * lda);
        lapack::larfg(i_n, &alpha, A + (std::min(i+2,n-1)) + (i) * lda, ione, &tau[i]);
        e[i] = alpha;
        *(A + (i+1) + i * lda) = MAGMA_D_ONE;

        /* Compute W(i+1:n,i) */
        // 1. Send the block reflector  A(i+1:n,i) to the GPU
        magma_dsetvector_async( i_n, A + (i+1) + (i) * lda, 1, dA, dA_offset + (i+1) + (i) * ldda, 1, queue );

        magmablas_dsymv_work( MagmaLower, i_n, c_one, dA, dA_offset + (i+1) + (i+1) * ldda, ldda,
                              dA, dA_offset + (i+1) + (i) * ldda, ione, c_zero, dW, dW_offset + (i+1) + (i) * lddw, ione,
                              dwork, dwork_offset, ldwork, queue );

        // 2. Start getting the result back (asynchronously)
        magma_dgetmatrix_async( i_n, 1,
                                dW, dW_offset + (i+1) + (i) * lddw, lddw,
                                W + (i+1) + (i) * ldw,  ldw, queue );

        blas::gemv('C', i_n, i,
                   c_one,     W + (i+1),             ldw,
                              A + (i+1) + (i) * lda, ione, c_zero,
                              W + (i) * ldw,         ione);
        blas::gemv('N', i_n, i,
                   c_neg_one, A + (i+1),             lda,
                              W + (i) * ldw,         ione, c_zero,
                              work,                  ione);
        blas::gemv('C', i_n, i,
                   c_one,     A + (i+1),             lda,
                              A + (i+1) + (i) * lda, ione, c_zero,
                              W + (i) * ldw,         ione);

        // 3. Here we need dsymv result W(i+1, i)
        magma_queue_sync( queue );

        if (i != 0)
          {
          blas::axpy(i_n, c_one, work, ione, W + (i+1) + (i) * ldw, ione);
          }

        blas::gemv('N', i_n, i,
                   c_neg_one, W + (i+1),             ldw,
                              W + (i) * ldw,         ione, c_one,
                              W + (i+1) + (i) * ldw, ione);
        blas::scal(i_n, tau[i], W + (i+1) + (i) * ldw, ione);

        value = magma_cblas_ddot( i_n, W + (i+1) + (i) * ldw, ione, A + (i+1) + (i) * lda, ione );
        alpha = tau[i] * -0.5f * value;
        blas::axpy(i_n, alpha, A + (i+1) + (i) * lda, ione, W + (i+1) + (i) * ldw, ione);
        }
      }
    }

  return info;
  } /* magma_dlatrd */
