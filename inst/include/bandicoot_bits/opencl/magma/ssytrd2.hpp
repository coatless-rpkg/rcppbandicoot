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
magma_get_ssytrd_nb( magma_int_t /* n */ )
  {
  return 64;
  }



// Purpose
// -------
// SSYTRD2_GPU reduces a real symmetric matrix A to real symmetric
// tridiagonal form T by an orthogonal similarity transformation:
// Q**H * A * Q = T.
// This version passes a workspace that is used in an optimized
// GPU matrix-vector product.



inline
magma_int_t
magma_ssytrd2_gpu
  (
  magma_uplo_t uplo, magma_int_t n,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  float *d, float *e, float *tau,
  float *A,  magma_int_t lda,
  float *work, magma_int_t lwork,
  magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const float c_zero    = MAGMA_S_ZERO;
  const float c_neg_one = MAGMA_S_NEG_ONE;
  const float c_one     = MAGMA_S_ONE;
  const float             d_one     = MAGMA_D_ONE;

  /* Local variables */
  const char* uplo_ = lapack_uplo_const( uplo );

  magma_int_t nb = magma_get_ssytrd_nb( n );

  magma_int_t kk, nx;
  magma_int_t i, j, i_n;
  magma_int_t iinfo;
  magma_int_t ldw, lddw, lwkopt;
  magma_int_t lquery;

  *info = 0;
  bool upper = (uplo == MagmaUpper);
  lquery = (lwork == -1);
  if (! upper && uplo != MagmaLower)
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (ldda < std::max(1,n))
    {
    *info = -4;
    }
  else if (lda < std::max(1,n))
    {
    *info = -9;
    }
  else if (lwork < nb*n && ! lquery)
    {
    *info = -11;
    }
  else if (ldwork < ldda*magma_ceildiv(n,64) + 2*ldda*nb)
    {
    *info = -13;
    }

  /* Determine the block size. */
  ldw = n;
  lddw = ldda;  // hopefully ldda is rounded up to multiple of 32; ldwork is in terms of ldda, so lddw can't be > ldda.
  lwkopt = n * nb;
  if (*info == 0)
    {
    work[0] = magma_smake_lwork( lwkopt );
    }

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
  if (n == 0)
    {
    work[0] = c_one;
    return *info;
    }

  // nx <= n is required
  // use LAPACK for n < 3000, otherwise switch at 512
  if (n < 3000)
    {
    nx = n;
    }
  else
    {
    nx = 512;
    }

  float *work2;
  if (MAGMA_SUCCESS != magma_smalloc_cpu( &work2, n ))
    {
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }

  magma_queue_t queue = magma_queue_create();

  // clear out dwork in case it has NANs (used as y in ssymv)
  // rest of dwork (used as work in magmablas_ssymv) doesn't need to be cleared
  magmablas_slaset( MagmaFull, n, nb, c_zero, c_zero, dwork, dwork_offset, lddw, queue );

  if (upper)
    {
    /* Reduce the upper triangle of A.
       Columns 1:kk are handled by the unblocked method. */
    kk = n - magma_roundup( n - nx, nb );

    for (i = n - nb; i >= kk; i -= nb)
      {
      /* Reduce columns i:i+nb-1 to tridiagonal form and form the
         matrix W which is needed to update the unreduced part of
         the matrix */

      /* Get the current panel */
      magma_sgetmatrix( i+nb, nb, dA, dA_offset + i * ldda, ldda, A + i * lda, lda, queue );

      magma_slatrd2( uplo, i+nb, nb, A, lda, e, tau,
                     work, ldw, work2, n, dA, dA_offset, ldda, dwork, dwork_offset, lddw,
                     dwork, dwork_offset + 2*lddw*nb, ldwork - 2*lddw*nb, queue );

      /* Update the unreduced submatrix A(0:i-2,0:i-2), using an
         update of the form:  A := A - V*W' - W*V' */
      magma_ssetmatrix( i + nb, nb, work, ldw, dwork, dwork_offset, lddw, queue );

      magmablas_ssyr2k( uplo, MagmaNoTrans, i, nb, c_neg_one,
                        dA, dA_offset + i * ldda, ldda, dwork, dwork_offset, lddw,
                        d_one, dA, dA_offset, ldda, queue );

      /* Copy superdiagonal elements back into A, and diagonal
         elements into D */
      for (j = i; j < i+nb; ++j)
        {
        *(A + (j-1) + j * lda) = e[j - 1];
        d[j] = *(A + j + j * lda);
        }
      }

    magma_sgetmatrix( kk, kk, dA, dA_offset, ldda, A, lda, queue );

    /* Use CPU code to reduce the last or only block */
    lapack::sytrd(uplo_[0], kk, A, lda, d, e, tau, work, lwork, &iinfo);

    magma_ssetmatrix( kk, kk, A, lda, dA, dA_offset, ldda, queue );
    }
  else
    {
    /* Reduce the lower triangle of A */
    for (i = 0; i < n-nx; i += nb)
      {
      /* Reduce columns i:i+nb-1 to tridiagonal form and form the
         matrix W which is needed to update the unreduced part of
         the matrix */

      /* Get the current panel */
      magma_sgetmatrix( n-i, nb, dA, dA_offset + i + i * ldda, ldda, A + i + i * lda, lda, queue );

      magma_slatrd2( uplo, n-i, nb, A + i + i * lda, lda, &e[i], &tau[i],
                     work, ldw, work2, n, dA, dA_offset + i + i * ldda, ldda, dwork, dwork_offset, lddw,
                     dwork, dwork_offset + 2*lddw*nb, ldwork - 2*lddw*nb, queue );

      /* Update the unreduced submatrix A(i+ib:n,i+ib:n), using
         an update of the form:  A := A - V*W' - W*V' */
      magma_ssetmatrix( n-i, nb, work, ldw, dwork, dwork_offset, lddw, queue );

      // cublas 6.5 crashes here if lddw % 32 != 0, e.g., N=250.
      magmablas_ssyr2k( MagmaLower, MagmaNoTrans, n-i-nb, nb, c_neg_one,
                        dA, dA_offset + (i+nb) + i * ldda, ldda, dwork, dwork_offset + nb, lddw,
                        d_one, dA, dA_offset + (i+nb) + (i+nb) * ldda, ldda, queue );

      /* Copy subdiagonal elements back into A, and diagonal
         elements into D */
      for (j = i; j < i+nb; ++j)
        {
        *(A + (j+1) + j * lda) = e[j];
        d[j] = *(A + j + j * lda);
        }
      }

    /* Use CPU code to reduce the last or only block */
    magma_sgetmatrix( n-i, n-i, dA, dA_offset + i + i * ldda, ldda, A + i + i * lda, lda, queue );

    i_n = n-i;
    lapack::sytrd(uplo_[0], i_n, A + i + i * lda, lda, &d[i], &e[i], &tau[i], work, lwork, &iinfo);

    magma_ssetmatrix( n-i, n-i, A + i + i * lda, lda, dA, dA_offset + i + i * ldda, ldda, queue );
    }

  magma_free_cpu( work2 );
  magma_queue_destroy( queue );

  work[0] = magma_smake_lwork( lwkopt );

  return *info;
  } /* magma_ssytrd2_gpu */
