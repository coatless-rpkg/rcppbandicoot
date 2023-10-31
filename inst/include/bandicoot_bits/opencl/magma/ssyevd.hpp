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



// Purpose
// -------
// SSYEVD_GPU computes all eigenvalues and, optionally, eigenvectors of
// a real symmetric matrix A.  If eigenvectors are desired, it uses a
// divide and conquer algorithm.
//
// The divide and conquer algorithm makes very mild assumptions about
// floating point arithmetic. It will work on machines with a guard
// digit in add/subtract, or on those binary machines without guard
// digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
// Cray-2. It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.



inline
magma_int_t
magma_ssyevd_gpu
  (
  magma_vec_t jobz, magma_uplo_t uplo,
  magma_int_t n,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  float *w,
  float *wA,  magma_int_t ldwa,
  float *work, magma_int_t lwork,
  magma_int_t *iwork, magma_int_t liwork,
  magma_int_t *info
  )
  {
  magma_int_t ione = 1;

  float d__1;

  float eps;
  magma_int_t inde;
  float anrm;
  float rmin, rmax;
  float sigma;
  magma_int_t iinfo, lwmin;
  magma_int_t lower;
  magma_int_t wantz;
  magma_int_t indwk2, llwrk2;
  magma_int_t iscale;
  float safmin;
  float bignum;
  magma_int_t indtau;
  magma_int_t indwrk, liwmin;
  magma_int_t llwork;
  float smlnum;
  magma_int_t lquery;

  magmaFloat_ptr dwork;
  magma_int_t lddc = ldda;

  wantz = (jobz == MagmaVec);
  lower = (uplo == MagmaLower);
  lquery = (lwork == -1 || liwork == -1);

  *info = 0;
  if (! (wantz || (jobz == MagmaNoVec)))
    {
    *info = -1;
    }
  else if (! (lower || (uplo == MagmaUpper)))
    {
    *info = -2;
    }
  else if (n < 0)
    {
    *info = -3;
    }
  else if (ldda < std::max(1,n))
    {
    *info = -5;
    }

  magma_int_t nb = magma_get_ssytrd_nb( n );
  if ( n <= 1 )
    {
    lwmin  = 1;
    liwmin = 1;
    }
  else if ( wantz )
    {
    lwmin  = std::max( 2*n + n*nb, 1 + 6*n + 2*n*n );
    liwmin = 3 + 5*n;
    }
  else
    {
    lwmin  = 2*n + n*nb;
    liwmin = 1;
    }

  work[0]  = magma_smake_lwork( lwmin );
  iwork[0] = liwmin;

  if ((lwork < lwmin) && !lquery)
    {
    *info = -10;
    }
  else if ((liwork < liwmin) && ! lquery)
    {
    *info = -12;
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

  magma_queue_t queue = magma_queue_create();

  /* If matrix is very small, then just call LAPACK on CPU, no need for GPU */
  if (n <= 128)
    {
    magma_int_t lda = n;
    float *A;
    magma_smalloc_cpu( &A, lda*n );
    magma_sgetmatrix( n, n, dA, dA_offset, ldda, A, lda, queue );
    lapack::syevd(lapack_vec_const(jobz)[0], lapack_uplo_const(uplo)[0],
                  n, A, lda,
                  w, work, lwork,
                  iwork, liwork, info);
    magma_ssetmatrix( n, n, A, lda, dA, dA_offset, ldda, queue );
    magma_free_cpu( A );
    magma_queue_destroy( queue );
    return *info;
    }

  // ssytrd2_gpu requires ldda*ceildiv(n,64) + 2*ldda*nb
  // sormtr_gpu  requires lddc*n
  // slansy      requires n
  magma_int_t ldwork = std::max( ldda*magma_ceildiv(n,64) + 2*ldda*nb, lddc*n );
  ldwork = std::max( ldwork, n );
  if ( wantz )
    {
    // sstedx requires 3n^2/2
    ldwork = std::max( ldwork, 3*n*(n/2 + 1) );
    }

  if (MAGMA_SUCCESS != magma_smalloc( &dwork, ldwork ))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }

  /* Get machine constants. */
  safmin = lapack::lamch<float>('S');
  eps    = lapack::lamch<float>('P');
  smlnum = safmin / eps;
  bignum = 1. / smlnum;
  rmin = std::sqrt( smlnum );
  rmax = std::sqrt( bignum );

  /* Scale matrix to allowable range, if necessary. */
  anrm = magmablas_slansy( MagmaMaxNorm, uplo, n, dA, dA_offset, ldda, dwork, 0, ldwork, queue );
  iscale = 0;
  sigma  = 1;
  if (anrm > 0. && anrm < rmin)
    {
    iscale = 1;
    sigma = rmin / anrm;
    }
  else if (anrm > rmax)
    {
    iscale = 1;
    sigma = rmax / anrm;
    }

  if (iscale == 1)
    {
    magmablas_slascl( uplo, 0, 0, 1., sigma, n, n, dA, dA_offset, ldda, queue, info );
    }

  /* Call SSYTRD to reduce symmetric matrix to tridiagonal form. */
  // ssytrd work: e (n) + tau (n) + llwork (n*nb)  ==>  2n + n*nb
  // sstedx work: e (n) + tau (n) + z (n*n) + llwrk2 (1 + 4*n + n^2)  ==>  1 + 6n + 2n^2
  inde   = 0;
  indtau = inde   + n;
  indwrk = indtau + n;
  indwk2 = indwrk + n*n;
  llwork = lwork - indwrk;
  llwrk2 = lwork - indwk2;

  magma_ssytrd2_gpu( uplo, n, dA, dA_offset, ldda, w, &work[inde],
                     &work[indtau], wA, ldwa, &work[indwrk], llwork,
                     dwork, 0, ldwork, &iinfo );

  /* For eigenvalues only, call SSTERF.  For eigenvectors, first call
     SSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
     tridiagonal matrix, then call SORMTR to multiply it to the Householder
     transformations represented as Householder vectors in A. */
  if (! wantz)
    {
    lapack::sterf(n, w, &work[inde], info);
    }
  else
    {
    magma_sstedx( MagmaRangeAll, n, 0., 0., 0, 0, w, &work[inde],
                  &work[indwrk], n, &work[indwk2],
                  llwrk2, iwork, liwork, dwork, 0, info );

    magma_ssetmatrix( n, n, &work[indwrk], n, dwork, 0, lddc, queue );

    magma_sormtr_gpu( MagmaLeft, uplo, MagmaNoTrans, n, n, dA, dA_offset, ldda, &work[indtau],
                      dwork, 0, lddc, wA, ldwa, &iinfo );

    magma_scopymatrix( n, n, dwork, 0, lddc, dA, dA_offset, ldda, queue );
    }

  /* If matrix was scaled, then rescale eigenvalues appropriately. */
  if (iscale == 1)
    {
    d__1 = 1. / sigma;
    blas::scal(n, d__1, w, ione);
    }

  work[0]  = magma_smake_lwork( lwmin );
  iwork[0] = liwmin;

  magma_queue_destroy( queue );
  magma_free( dwork );

  return *info;
  } /* magma_ssyevd_gpu */
