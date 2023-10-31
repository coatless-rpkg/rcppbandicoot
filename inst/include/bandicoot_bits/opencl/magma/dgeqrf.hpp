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
magma_get_dgeqrf_nb( magma_int_t m, magma_int_t n )
  {
  // Note: this is tuned for AMD Tahiti cards (taken from clBLAS).
  magma_int_t minmn = std::min(m, n);

  if   (minmn <= 2048) return  64;
  else                 return 128;
  }



inline
void
dsplit_diag_block_invert
  (
  magma_int_t ib,
  double *A,
  magma_int_t lda,
  double *work
  )
  {
  const double c_zero = MAGMA_D_ZERO;
  const double c_one  = MAGMA_D_ONE;

  magma_int_t i, j, info;
  double *cola, *colw;

  for (i = 0; i < ib; i++)
    {
    cola = A    + i*lda;
    colw = work + i*ib;
    for (j = 0; j < i; j++)
      {
      colw[j] = cola[j];
      cola[j] = c_zero;
      }
    colw[i] = cola[i];
    cola[i] = c_one;
    }

  lapack::trtri(MagmaUpperStr[0], MagmaNonUnitStr[0], ib, work, ib, &info);
  }



// DGEQRF computes a QR factorization of a real M-by-N matrix A:
// A = Q * R.
//
// This version stores the triangular dT matrices used in
// the block QR factorization so that they can be applied directly (i.e.,
// without being recomputed) later. As a result, the application
// of Q is much faster. Also, the upper triangular matrices for V have 0s
// in them. The corresponding parts of the upper triangular R are inverted and
// stored separately in dT.



inline
magma_int_t
magma_dgeqrf_gpu
  (
  magma_int_t m, magma_int_t n,
  magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
  double *tau,
  magmaDouble_ptr dT, size_t dT_offset,
  magma_int_t *info
  )
  {
  double *work, *hwork, *R;
  magma_int_t cols, i, ib, ldwork, lddwork, lhwork, lwork, minmn, nb, old_i, old_ib, rows;

  // check arguments
  *info = 0;
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (ldda < std::max(1,m))
    *info = -4;

  if (*info != 0)
    {
    // magma_xerbla( __func__, -(*info) );
    return *info;
    }

  minmn = std::min( m, n );
  if (minmn == 0)
    return *info;

  // TODO: use min(m,n), but that affects dT
  nb = magma_get_dgeqrf_nb( m, n );

  // dT contains 3 blocks:
  // dT    is minmn*nb
  // dR    is minmn*nb
  // dwork is n*nb
  magmaDouble_ptr dR = dT;
  size_t dR_offset = dT_offset + minmn;
  magmaDouble_ptr dwork = dT;
  size_t dwork_offset = dT_offset + 2 * minmn;
  lddwork = n;

  // work  is m*nb for panel
  // hwork is n*nb, and at least nb*nb for T in larft
  // R     is nb*nb
  ldwork = m;
  lhwork = std::max( n*nb, nb*nb );
  lwork  = ldwork*nb + lhwork + nb*nb;
  // last block needs rows*cols for matrix and prefers cols*nb for work
  // worst case is n > m*nb, m a small multiple of nb:
  // needs n*nb + n > (m+n)*nb
  // prefers 2*n*nb, about twice above (m+n)*nb.
  i = ((minmn-1)/nb)*nb;
  lwork = std::max( lwork, (m-i)*(n-i) + (n-i)*nb );

  if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lwork ))
    {
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }
  hwork = work + ldwork*nb;
  R     = work + ldwork*nb + lhwork;
  std::memset( R, 0, nb*nb*sizeof(double) );

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if ( nb > 1 && nb < minmn )
    {
    // need nb*nb for T in larft
    assert( lhwork >= nb*nb );

    // Use blocked code initially
    old_i = 0; old_ib = nb;
    for (i = 0; i < minmn-nb; i += nb)
      {
      ib = std::min( minmn-i, nb );
      rows = m - i;

      // get i-th panel from device
      magma_dgetmatrix_async( rows, ib,
                              dA, i + i * ldda, ldda,
                              work,    ldwork, queues[1] );
      if (i > 0)
        {
        // Apply H^H to A(i:m,i+2*ib:n) from the left
        cols = n - old_i - 2*old_ib;
        magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                          m-old_i, cols, old_ib,
                          dA,    dA_offset + old_i + old_i * ldda,                ldda,
                          dT,    dT_offset + old_i * nb,                          nb,
                          dA,    dA_offset + old_i + (old_i + 2 * old_ib) * ldda, ldda,
                          dwork, dwork_offset,                                    lddwork, queues[0] );

        // Fix the diagonal block
        magma_dsetmatrix_async( old_ib, old_ib,
                                R,                          old_ib,
                                dR, dR_offset + old_i * nb, old_ib, queues[0] );
        }

      magma_queue_sync( queues[1] );  // wait to get work(i)
      lapack::geqrf(rows, ib, work, ldwork, &tau[i], hwork, lhwork, info);

      // Form the triangular factor of the block reflector in hwork
      // H = H(i) H(i+1) . . . H(i+ib-1)
      lapack::larft(MagmaForwardStr[0], MagmaColumnwiseStr[0],
                    rows, ib,
                    work, ldwork, &tau[i], hwork, ib);

      // wait for previous trailing matrix update (above) to finish with R
      magma_queue_sync( queues[0] );

      // copy the upper triangle of panel to R and invert it, and
      // set  the upper triangle of panel (V) to identity
      dsplit_diag_block_invert( ib, work, ldwork, R );

      // send i-th V matrix to device
      magma_dsetmatrix( rows, ib,
                        work, ldwork,
                        dA, dA_offset + i + i * ldda, ldda, queues[1] );

      if (i + ib < n)
        {
        // send T matrix to device
        magma_dsetmatrix( ib, ib,
                          hwork, ib,
                          dT, dT_offset + i * nb, nb, queues[1] );

        if (i+nb < minmn-nb)
          {
          // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, ib, ib,
                            dA,    dA_offset + i + i * ldda,        ldda,
                            dT,    dT_offset + i * nb,              nb,
                            dA,    dA_offset + i + (i + ib) * ldda, ldda,
                            dwork, dwork_offset,                    lddwork, queues[1] );
          // wait for larfb to finish with dwork before larfb in next iteration starts
          magma_queue_sync( queues[1] );
          }
        else
          {
          // Apply H^H to A(i:m,i+ib:n) from the left
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, n-i-ib, ib,
                            dA,    dA_offset + i + i * ldda,        ldda,
                            dT,    dT_offset + i * nb,              nb,
                            dA,    dA_offset + i + (i + ib) * ldda, ldda,
                            dwork, dwork_offset,                    lddwork, queues[1] );
          // Fix the diagonal block
          magma_dsetmatrix( ib, ib,
                            R,     ib,
                            dR, dR_offset + i * nb, ib, queues[1] );
          }
        old_i  = i;
        old_ib = ib;
        }
      }
    }
  else
    {
    i = 0;
    }

  // Use unblocked code to factor the last or only block.
  if (i < minmn)
    {
    rows = m-i;
    cols = n-i;
    magma_dgetmatrix( rows, cols, dA, dA_offset + i + i * ldda, ldda, work, rows, queues[1] );
    // see comments for lwork above
    lhwork = lwork - rows*cols;
    lapack::geqrf(rows, cols, work, rows, &tau[i], &work[rows * cols], lhwork, info);

    magma_dsetmatrix( rows, cols, work, rows, dA, dA_offset + i + i * ldda, ldda, queues[1] );
    }

  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  magma_free_pinned( work );

  return *info;
  } // magma_dgeqrf_gpu



// DGEQRF computes a QR factorization of a DOUBLE PRECISION M-by-N matrix A:
// A = Q * R. This version does not require work space on the GPU
// passed as input. GPU memory is allocated in the routine.



inline
magma_int_t
magma_dgeqrf
  (
  magma_int_t m, magma_int_t n,
  double *A,    magma_int_t lda,
  double *tau,
  double *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one = MAGMA_D_ONE;

  /* Local variables */
  double* work_local = NULL;
  magmaDouble_ptr dA, dT, dwork;
  size_t dT_offset, dwork_offset;
  magma_int_t i, ib, min_mn, ldda, lddwork, old_i, old_ib;

  /* Function Body */
  *info = 0;
  magma_int_t nb = magma_get_dgeqrf_nb( m, n );

  magma_int_t lwkopt = n*nb;
  work[0] = magma_dmake_lwork( lwkopt );
  bool lquery = (lwork == -1);
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (lda < std::max(1, m))
    *info = -4;
  else if (lwork < std::max(1, lwkopt) && !lquery)
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

  min_mn = std::min( m, n );
  if (min_mn == 0)
    {
    work[0] = c_one;
    return *info;
    }

  if (nb <= 1 || 4*nb >= std::min(m,n) )
    {
    /* Use CPU code. */
    lapack::geqrf(m, n, A, lda, tau, work, lwork, info);
    return *info;
    }

  // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
  lddwork = magma_roundup( n, 32 ) - nb;
  ldda    = magma_roundup( m, 32 );

  // allocate space for dA, dwork, and dT
  if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + nb*lddwork + nb*nb ))
    {
    /* alloc failed so call non-GPU-resident version */
    // TODO: port the function below
    //return magma_dgeqrf_ooc( m, n, A, lda, tau, work, lwork, info );
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }

  // Need at least 2*nb*nb to store T and upper triangle of V simultaneously.
  // For better LAPACK compatability, which needs N*NB,
  // allow lwork < 2*NB*NB and allocate here if needed.
  if (lwork < 2*nb*nb)
    {
    if (MAGMA_SUCCESS != magma_dmalloc_cpu( &work_local, 2*nb*nb ))
      {
      magma_free( dA );
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
      }
    work = work_local;
    }

  dwork = dA;
  dwork_offset = n*ldda;
  dT    = dA;
  dT_offset = n*ldda + nb*lddwork;

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if ( (nb > 1) && (nb < min_mn) )
    {
    /* Use blocked code initially.
       Asynchronously send the matrix to the GPU except the first panel. */
    magma_dsetmatrix_async( m, n-nb,
                             &A[nb * lda], lda,
                            dA, nb * ldda, ldda, queues[0] );

    old_i = 0;
    old_ib = nb;
    for (i = 0; i < min_mn-nb; i += nb)
      {
      ib = std::min( min_mn-i, nb );
      if (i > 0)
        {
        /* get i-th panel from device */
        magma_queue_sync( queues[1] );
        magma_dgetmatrix_async( m-i, ib,
                                dA, i + i * ldda, ldda,
                                 &A[i + i * lda], lda, queues[0] );

        /* Apply H' to A(i:m,i+2*ib:n) from the left */
        magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                          m-old_i, n-old_i-2*old_ib, old_ib,
                          dA,    old_i + old_i * ldda,            ldda,
                          dT,    dT_offset,                       nb,
                          dA,    old_i + (old_i+2*old_ib) * ldda, ldda,
                          dwork, dwork_offset,                    lddwork, queues[1] );

        magma_dgetmatrix_async( i, ib,
                                dA, i * ldda, ldda,
                                 &A[i * lda], lda, queues[1] );
        magma_queue_sync( queues[0] );
        }

      magma_int_t rows = m-i;
      lapack::geqrf(rows, ib, &A[i + i * lda], lda, &tau[i], work, lwork, info);

      /* Form the triangular factor of the block reflector
         H = H(i) H(i+1) . . . H(i+ib-1) */
      lapack::larft(MagmaForwardStr[0], MagmaColumnwiseStr[0],
                    rows, ib, &A[i + i * lda], lda, &tau[i], work, ib);

      magma_dpanel_to_q( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );

      /* put i-th V matrix onto device */
      magma_dsetmatrix_async( rows, ib, &A[i + i * lda], lda, dA, i + i * ldda, ldda, queues[0] );

      /* put T matrix onto device */
      magma_queue_sync( queues[1] );
      magma_dsetmatrix_async( ib, ib, work, ib, dT, dT_offset, nb, queues[0] );
      magma_queue_sync( queues[0] );

      if (i + ib < n)
        {
        if (i+ib < min_mn-nb)
          {
          /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, ib, ib,
                            dA,    i + i * ldda,      ldda,
                            dT,    dT_offset,         nb,
                            dA,    i + (i+ib) * ldda, ldda,
                            dwork, dwork_offset,      lddwork, queues[1] );
          magma_dq_to_panel( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );
          }
        else
          {
          /* After last panel, update whole trailing matrix. */
          /* Apply H' to A(i:m,i+ib:n) from the left */
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, n-i-ib, ib,
                            dA,    i + i * ldda,      ldda,
                            dT,    dT_offset,         nb,
                            dA,    i + (i+ib) * ldda, ldda,
                            dwork, dwork_offset,      lddwork, queues[1] );
          magma_dq_to_panel( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );
          }

        old_i  = i;
        old_ib = ib;
        }
      }
    }
  else
    {
    i = 0;
    }

  /* Use unblocked code to factor the last or only block. */
  if (i < min_mn)
    {
    ib = n-i;
    if (i != 0)
      {
      magma_dgetmatrix( m, ib, dA, i * ldda, ldda, &A[i * lda], lda, queues[1] );
      }
    magma_int_t rows = m-i;
    lapack::geqrf(rows, ib, &A[i + i * lda], lda, &tau[i], work, lwork, info);
    }

  magma_queue_sync( queues[0] );
  magma_queue_sync( queues[1] );
  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  work[0] = magma_dmake_lwork( lwkopt );  // before free( work_local )

  magma_free( dA );
  magma_free_cpu( work_local );  // if allocated

  return *info;
  } /* magma_dgeqrf */
