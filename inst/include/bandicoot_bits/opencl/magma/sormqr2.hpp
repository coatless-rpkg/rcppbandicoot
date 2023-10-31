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
// SORMQR overwrites the general real M-by-N matrix C with
//
// @verbatim
//                            SIDE = MagmaLeft    SIDE = MagmaRight
// TRANS = MagmaNoTrans:      Q * C               C * Q
// TRANS = MagmaTrans:   Q**H * C            C * Q**H
// @endverbatim
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by SGEQRF.
// Q is of order M if SIDE = MagmaLeft
// and  of order N if SIDE = MagmaRight.



inline
magma_int_t
magma_sormqr2_gpu
  (
  magma_side_t side, magma_trans_t trans,
  magma_int_t m, magma_int_t n, magma_int_t k,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  float    *tau,
  magmaFloat_ptr dC, size_t dC_offset, magma_int_t lddc,
  float *wA, magma_int_t ldwa,
  magma_int_t *info
  )
  {
  /* Constants */
  const float c_zero = MAGMA_S_ZERO;
  const float c_one  = MAGMA_S_ONE;
  const magma_int_t nbmax = 64;

  /* Local variables */
  magmaFloat_ptr dwork = NULL, dT = NULL;
  float T[ nbmax*nbmax ];
  magma_int_t i, i1, i2, step, ib, ic, jc, lddwork, nb, mi, ni, nq, nq_i, nw;
  magma_queue_t queue = NULL;

  // Parameter adjustments for Fortran indexing
  wA -= 1 + ldwa;
  size_t dC_neg_offset = 1 + lddc;
  --tau;

  *info = 0;
  bool left   = (side == MagmaLeft);
  bool notran = (trans == MagmaNoTrans);

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

  if (! left && side != MagmaRight)
    {
    *info = -1;
    }
  else if (! notran && trans != MagmaTrans)
    {
    *info = -2;
    }
  else if (m < 0)
    {
    *info = -3;
    }
  else if (n < 0)
    {
    *info = -4;
    }
  else if (k < 0 || k > nq)
    {
    *info = -5;
    }
  else if (ldda < std::max(1,nq))
    {
    *info = -7;
    }
  else if (lddc < std::max(1,m))
    {
    *info = -10;
    }
  else if (ldwa < std::max(1,nq))
    {
    *info = -12;
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  /* Quick return if possible */
  if (m == 0 || n == 0 || k == 0)
    {
    return *info;
    }

  // size of the block
  nb = nbmax;

  lddwork = nw;

  /* Use hybrid CPU-GPU code */
  if ( (  left && ! notran) ||
       (! left &&   notran) )
    {
    i1 = 1;
    i2 = k;
    step = nb;
    }
  else
    {
    i1 = ((k - 1)/nb)*nb + 1;
    i2 = 1;
    step = -nb;
    }

  // silence "uninitialized" warnings
  mi = 0;
  ni = 0;

  if (left)
    {
    ni = n;
    jc = 1;
    }
  else
    {
    mi = m;
    ic = 1;
    }

  // dwork is (n or m) x nb + nb x nb, for left or right respectively
  if (MAGMA_SUCCESS != magma_smalloc( &dwork, lddwork*nb + nb*nb ))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    magma_queue_destroy( queue );
    return *info;
    }

  dT = dwork;
  size_t dT_offset = lddwork*nb;

  queue = magma_queue_create();

  // set nb-1 super-diagonals to 0, and diagonal to 1.
  // This way we can copy V directly to the GPU,
  // with the upper triangle parts already set to identity.
  magmablas_slaset_band( MagmaUpper, k, k, nb, c_zero, c_one, dA, dA_offset, ldda, queue );

  // for i=i1 to i2 by step
  for (i = i1; (step < 0 ? i >= i2 : i <= i2); i += step)
    {
    ib = std::min( nb, k - i + 1 );

    /* Form the triangular factor of the block reflector
       H = H(i) H(i+1) . . . H(i+ib-1) */
    nq_i = nq - i + 1;
    lapack::larft('F', 'C', nq_i, ib,
                  wA + (i) + (i) * ldwa, ldwa, &tau[i], T, ib);

    if (left)
      {
      /* H or H^H is applied to C(i:m,1:n) */
      mi = m - i + 1;
      ic = i;
      }
    else
      {
      /* H or H^H is applied to C(1:m,i:n) */
      ni = n - i + 1;
      jc = i;
      }

    /* Apply H or H^H; First copy T to the GPU */
    magma_ssetmatrix( ib, ib, T, ib, dT, dT_offset, ib, queue );
    magma_slarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                      mi, ni, ib,
                      dA, dA_offset + (i-1) + (i-1) * ldda, ldda, dT, dT_offset, ib,  // dA using 0-based indices here
                      dC, dC_offset + (ic) + (jc) * lddc - dC_neg_offset, lddc,
                      dwork, 0, lddwork, queue );
    }

  magma_queue_destroy( queue );
  magma_free( dwork );

  return *info;
  } /* magma_sormqr */
