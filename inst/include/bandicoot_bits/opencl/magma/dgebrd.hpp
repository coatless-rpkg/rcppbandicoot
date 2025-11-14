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



// DGEBRD reduces a general real M-by-N matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**H * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

inline
magma_int_t
magma_get_dgebrd_nb(magma_int_t /* m */, magma_int_t /* n */)
  {
  return 32;
  }



inline
magma_int_t
magma_dgebrd
  (
  magma_int_t m, magma_int_t n,
  double *A, magma_int_t lda, double *d, double *e,
  double *tauq, double *taup,
  double *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  double c_neg_one = MAGMA_D_NEG_ONE;
  double c_one     = MAGMA_D_ONE;
  magmaDouble_ptr dA, dwork;

  magma_int_t ncol, nrow, jmax, nb, ldda;

  magma_int_t i, j, nx;
  magma_int_t iinfo;

  magma_int_t minmn;
  magma_int_t ldwrkx, ldwrky, lwkopt;
  magma_int_t lquery;

  nb   = magma_get_dgebrd_nb( m, n );
  ldda = m;

  lwkopt = (m + n) * nb;
  work[0] = magma_dmake_lwork( lwkopt );
  lquery = (lwork == -1);

  /* Check arguments */
  *info = 0;
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (lda < std::max(1,m))
    *info = -4;
  else if (lwork < lwkopt && (! lquery) )
    *info = -10;

  if (*info < 0)
    {
    // magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;\
    }

  /* Quick return if possible */
  minmn = std::min(m,n);
  if (minmn == 0)
    {
    work[0] = c_one;
    return *info;
    }

  magma_queue_t queue = magma_queue_create();

  double *work2;
  magma_int_t lwork2 = std::max(m,n);
  if (MAGMA_SUCCESS != magma_dmalloc_cpu( &work2, lwork2 ))
    {
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }
  if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + (m + n)*nb ))
    {
    magma_free_cpu( work2 );
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }
  dwork = dA;
  size_t dwork_offset = n*ldda;

  ldwrkx = m;
  ldwrky = n;

  /* Set the block/unblock crossover point NX. */
  nx = 128;

  /* Copy the matrix to the GPU */
  if (minmn - nx >= 1)
    {
    magma_dsetmatrix( m, n, A, lda, dA, 0, ldda, queue );
    }

  for (i = 0; i < (minmn - nx); i += nb)
    {
    /*  Reduce rows and columns i:i+nb-1 to bidiagonal form and return
        the matrices X and Y which are needed to update the unreduced
        part of the matrix */
    nrow = m - i;
    ncol = n - i;

    /* Get the current panel (no need for the 1st iteration) */
    if ( i > 0 )
      {
      magma_dgetmatrix( nrow, nb,
                        dA, i + i * ldda, ldda,
                        &A[i + i * lda], lda, queue );
      magma_dgetmatrix( nb, ncol - nb,
                        dA, i + (i+nb) * ldda, ldda,
                        &A[i + (i+nb) * lda], lda, queue );
      }

    magma_dlabrd_gpu(nrow, ncol, nb,
                     &A[i + i * lda],  lda,    dA, i + (i * ldda),          ldda,
                     d+i, e+i, tauq+i, taup+i,
                     work,             ldwrkx, dwork, dwork_offset,               ldwrkx,  // x, dx
                     work+(ldwrkx*nb), ldwrky, dwork, dwork_offset + (ldwrkx*nb), ldwrky,
                     work2, lwork2, queue ); // y, dy

    /*  Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
        of the form  A := A - V*Y' - X*U' */
    nrow = m - i - nb;
    ncol = n - i - nb;

    // Send Y back to the GPU
    magma_dsetmatrix( nrow, nb,
                      work + nb, ldwrkx,
                      dwork, dwork_offset + nb, ldwrkx, queue );
    magma_dsetmatrix( ncol, nb,
                      work + (ldwrkx+1)*nb,                ldwrky,
                      dwork, dwork_offset + (ldwrkx+1)*nb, ldwrky, queue );

    magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                 nrow, ncol, nb,
                 c_neg_one, dA,    (i+nb) + i * ldda,                     ldda,
                            dwork, dwork_offset + (ldwrkx+1) * nb,        ldwrky,
                 c_one,     dA,    (i+nb) + (i+nb) * ldda,                ldda, queue );

    magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                 nrow, ncol, nb,
                 c_neg_one, dwork, dwork_offset + nb,   ldwrkx,
                            dA,      i + (i+nb) * ldda, ldda,
                 c_one,     dA, (i+nb) + (i+nb) * ldda, ldda, queue );

    /* Copy diagonal and off-diagonal elements of B back into A */
    if (m >= n)
      {
      jmax = i + nb;
      for (j = i; j < jmax; ++j)
        {
        A[j +     j * lda] = d[j];
        A[j + (j+1) * lda] = e[j];
        }
      }
    else
      {
      jmax = i + nb;
      for (j = i; j < jmax; ++j)
        {
        A[j +   j * lda] = d[j];
        A[j+1 + j * lda] = e[j];
        }
      }
    }

  /* Use unblocked code to reduce the remainder of the matrix */
  nrow = m - i;
  ncol = n - i;

  if ( 0 < minmn - nx )
    {
    magma_dgetmatrix( nrow, ncol,
                      dA, i + i * ldda, ldda,
                      &A[i + i * lda], lda, queue );
    }

  lapack::gebrd(nrow, ncol,
                &A[i + i * lda], lda, d+i, e+i,
                tauq+i, taup+i, work, lwork, &iinfo);
  work[0] = magma_dmake_lwork( lwkopt );

  magma_free_cpu( work2 );
  magma_free( dA );

  magma_queue_destroy( queue );

  return *info;
  } /* magma_dgebrd */
