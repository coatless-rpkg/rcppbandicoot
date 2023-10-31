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
magma_sormlq
  (
  magma_side_t side, magma_trans_t trans,
  magma_int_t m, magma_int_t n, magma_int_t k,
  float *A, magma_int_t lda,
  float *tau,
  float *C, magma_int_t ldc,
  float *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  float *T, *T2;
  magma_int_t i, i1, i2, ib, ic, jc, nb, mi, ni, nq, nq_i, nw, step;
  magma_int_t iinfo, ldwork, lwkopt;
  magma_trans_t transt;

  *info = 0;
  bool left   = (side  == MagmaLeft);
  bool notran = (trans == MagmaNoTrans);
  bool lquery = (lwork == -1);

  /* NQ is the order of Q and NW is the minimum dimension of WORK */
  if (left)
    {
    nq = m;
    nw = n;
    }
  else
    {
    nq = n;
    nw = m;
    }

  /* Test the input arguments */
  if (!left && side != MagmaRight)
    *info = -1;
  else if (!notran && trans != MagmaTrans)
    *info = -2;
  else if (m < 0)
    *info = -3;
  else if (n < 0)
    *info = -4;
  else if (k < 0 || k > nq)
    *info = -5;
  else if (lda < std::max(1,k))
    *info = -7;
  else if (ldc < std::max(1,m))
    *info = -10;
  else if (lwork < std::max(1,nw) && !lquery)
    *info = -12;

  if (*info == 0)
    {
    nb = magma_get_sgelqf_nb( m, n );
    lwkopt = std::max(1, nw) * nb;
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
  if (m == 0 || n == 0 || k == 0)
    {
    work[0] = MAGMA_D_ONE;
    return *info;
    }

  ldwork = nw;

  if (nb >= k)
    {
    /* Use CPU code */
    lapack::ormlq(lapack_side_const(side)[0], lapack_trans_const(trans)[0],
                  m, n, k, A, lda, tau, C, ldc, work, lwork, &iinfo);
    }
  else
    {
    /* Use hybrid CPU-GPU code */
    /* Allocate work space on the GPU.
     * nw*nb  for dwork (m or n) by nb
     * nq*nb  for dV    (n or m) by nb
     * nb*nb  for dT
     * lddc*n for dC.
     */
    magma_int_t lddc = magma_roundup( m, 32 );
    magmaFloat_ptr dwork, dV, dT, dC;
    size_t dV_offset, dT_offset, dC_offset;
    magma_smalloc( &dwork, (nw + nq + nb) * nb + lddc * n );
    if ( dwork == NULL )
      {
      *info = MAGMA_ERR_DEVICE_ALLOC;
      return *info;
      }
    dV = dwork;
    dV_offset = nw*nb;
    dT = dwork;
    dT_offset = dV_offset + nq * nb;
    dC = dwork;
    dC_offset = dT_offset + nb * nb;

    /* work space on CPU.
     * nb*nb for T
     * nb*nb for T2, used to save and restore diagonal block of panel  */
    magma_smalloc_cpu( &T, 2 * nb * nb );
    if ( T == NULL )
      {
      magma_free( dwork );
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
      }
    T2 = T + nb*nb;

    magma_queue_t queue = magma_queue_create();

    /* Copy matrix C from the CPU to the GPU */
    magma_ssetmatrix( m, n, C, ldc, dC, dC_offset, lddc, queue );

    if ( (left && notran) || (! left && !notran) )
      {
      i1 = 0;
      i2 = k;
      step = nb;
      }
    else
      {
      i1 = ((k - 1) / nb)*nb;
      i2 = 0;
      step = -nb;
      }

    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;

    if (left)
      {
      ni = n;
      jc = 0;
      }
    else
      {
      mi = m;
      ic = 0;
      }

    if (notran)
      {
      transt = MagmaTrans;
      }
    else
      {
      transt = MagmaNoTrans;
      }

    for (i = i1; (step < 0 ? i >= i2 : i < i2); i += step)
      {
      ib = std::min(nb, k - i);

      /* Form the triangular factor of the block reflector
         H = H(i) H(i + 1) . . . H(i + ib-1) */
      nq_i = nq - i;
      lapack::larft('F', 'R', nq_i, ib,
                    &A[i + i * lda], lda, &tau[i], T, ib);

      /* 1) set upper triangle of panel in A to identity,
         2) copy the panel from A to the GPU, and
         3) restore A                                      */
      magma_spanel_to_q( MagmaLower, ib, &A[i + i * lda], lda, T2 );
      magma_ssetmatrix( ib, nq_i,  &A[i + i * lda], lda, dV, dV_offset, ib, queue );
      magma_sq_to_panel( MagmaLower, ib, &A[i + i * lda], lda, T2 );

      if (left)
        {
        /* H or H**H is applied to C(i:m,1:n) */
        mi = m - i;
        ic = i;
        }
      else
        {
        /* H or H**H is applied to C(1:m,i:n) */
        ni = n - i;
        jc = i;
        }

      /* Apply H or H**H; First copy T to the GPU */
      magma_ssetmatrix( ib, ib, T, ib, dT, dT_offset, ib, queue );
      magma_slarfb_gpu( side, transt, MagmaForward, MagmaRowwise,
                        mi, ni, ib,
                        dV,    dV_offset,                  ib,
                        dT,    dT_offset,                  ib,
                        dC,    dC_offset + ic + jc * lddc, lddc,
                        dwork, 0,                          ldwork, queue );
    }
    magma_sgetmatrix( m, n, dC, dC_offset, lddc, C, ldc, queue );

    magma_queue_destroy( queue );
    magma_free( dwork );
    magma_free_cpu( T );
    }
  work[0] = magma_smake_lwork( lwkopt );

  return *info;
  } /* magma_sormlq */
