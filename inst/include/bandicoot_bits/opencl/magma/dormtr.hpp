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
// DORMTR overwrites the general real M-by-N matrix C with
//
//                             SIDE = MagmaLeft    SIDE = MagmaRight
// TRANS = MagmaNoTrans:       Q * C               C * Q
// TRANS = MagmaTrans:    Q**H * C            C * Q**H
//
// where Q is a real orthogonal matrix of order nq,
// with nq = m if SIDE = MagmaLeft
// and  nq = n if SIDE = MagmaRight. Q is defined as the product of
// nq-1 elementary reflectors, as returned by DSYTRD:
//
// if UPLO = MagmaUpper, Q = H(nq-1) . . . H(2) H(1);
// if UPLO = MagmaLower, Q = H(1) H(2) . . . H(nq-1).



inline
magma_int_t
magma_dormtr_gpu
  (
  magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
  magma_int_t m, magma_int_t n,
  magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
  double   *tau,
  magmaDouble_ptr dC, size_t dC_offset, magma_int_t lddc,
  double *wA, magma_int_t ldwa,
  magma_int_t *info
  )
  {
  magma_int_t i1, i2, mi, ni, nq;
  magma_int_t iinfo;

  *info = 0;
  bool left   = (side == MagmaLeft);
  bool upper  = (uplo == MagmaUpper);

  /* NQ is the order of Q and NW is the minimum dimension of WORK */
  if (left)
    {
    nq = m;
    //nw = n;
    }
  else
    {
    nq = n;
    //nw = m;
    }

  if (! left && side != MagmaRight)
    {
    *info = -1;
    }
  else if (! upper && uplo != MagmaLower)
    {
    *info = -2;
    }
  else if (trans != MagmaNoTrans &&
           trans != MagmaTrans)
    {
    *info = -3;
    }
  else if (m < 0)
    {
    *info = -4;
    }
  else if (n < 0)
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
  if (m == 0 || n == 0 || nq == 1)
    {
    return *info;
    }

  if (left)
    {
    mi = m - 1;
    ni = n;
    }
  else
    {
    mi = m;
    ni = n - 1;
    }

  if (upper)
    {
    magma_dormql2_gpu(side, trans, mi, ni, nq-1, dA, dA_offset + ldda, ldda, tau,
                      dC, dC_offset, lddc, wA + ldwa, ldwa, &iinfo);
    }
  else
    {
    /* Q was determined by a call to DSYTRD with UPLO = MagmaLower */
    if (left)
      {
      i1 = 1;
      i2 = 0;
      }
    else
      {
      i1 = 0;
      i2 = 1;
      }

    magma_dormqr2_gpu(side, trans, mi, ni, nq-1, dA, dA_offset + 1, ldda, tau,
                      dC, dC_offset + i1 + i2 * lddc, lddc, wA + 1, ldwa, &iinfo);
    }

  return *info;
  } /* magma_dormtr */
