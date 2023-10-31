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



// SLARFB applies a real block reflector H or its transpose H^H to a
// REAL m by n matrix C, from the left.



inline
magma_int_t
magma_slarfb_gpu
  (
  magma_side_t side,
  magma_trans_t trans,
  magma_direct_t direct,
  magma_storev_t storev,
  magma_int_t m,
  magma_int_t n,
  magma_int_t k,
  magmaFloat_const_ptr dV, size_t dV_offset,    magma_int_t lddv,
  magmaFloat_const_ptr dT, size_t dT_offset,    magma_int_t lddt,
  magmaFloat_ptr dC,       size_t dC_offset,    magma_int_t lddc,
  magmaFloat_ptr dwork,    size_t dwork_offset, magma_int_t ldwork,
  magma_queue_t queue
  )
  {
  /* Constants */
  const float c_zero    = MAGMA_S_ZERO;
  const float c_one     = MAGMA_S_ONE;
  const float c_neg_one = MAGMA_S_NEG_ONE;

  /* Check input arguments */
  magma_int_t info = 0;
  if (m < 0)
    {
    info = -5;
    }
  else if (n < 0)
    {
    info = -6;
    }
  else if (k < 0)
    {
    info = -7;
    }
  else if ( ((storev == MagmaColumnwise) && (side == MagmaLeft) && lddv < std::max(1,m)) ||
            ((storev == MagmaColumnwise) && (side == MagmaRight) && lddv < std::max(1,n)) ||
            ((storev == MagmaRowwise) && lddv < k) )
    {
    info = -9;
    }
  else if (lddt < k)
    {
    info = -11;
    }
  else if (lddc < std::max(1,m))
    {
    info = -13;
    }
  else if ( ((side == MagmaLeft) && ldwork < std::max(1,n)) ||
              ((side == MagmaRight) && ldwork < std::max(1,m)) )
    {
    info = -15;
    }

  if (info != 0)
    {
    // magma_xerbla( __func__, -(info) );
    return info;
    }

  /* Function Body */
  if (m <= 0 || n <= 0)
    {
    return info;
    }

  /* Local variables */
  // opposite of trans
  magma_trans_t transt;
  if (trans == MagmaNoTrans)
    transt = MagmaTrans;
  else
    transt = MagmaNoTrans;

  // whether T is upper or lower triangular
  magma_uplo_t uplo;
  if (direct == MagmaForward)
    uplo = MagmaUpper;
  else
    uplo = MagmaLower;

  // whether V is stored transposed or not
  magma_trans_t notransV, transV;
  if (storev == MagmaColumnwise)
    {
    notransV = MagmaNoTrans;
    transV   = MagmaTrans;
    }
  else
    {
    notransV = MagmaTrans;
    transV   = MagmaNoTrans;
    }

  if ( side == MagmaLeft )
    {
    // Form H C or H^H C
    // Comments assume H C.
    // When forming H^H C, T gets transposed via transt.

    // W = C^H V
    magma_sgemm( MagmaTrans, notransV,
                 n, k, m,
                 c_one,  dC, dC_offset,  lddc,
                         dV, dV_offset,  lddv,
                 c_zero, dwork, dwork_offset, ldwork, queue );

    // W = W T^H = C^H V T^H
    magma_strmm( MagmaRight, uplo, transt, MagmaNonUnit,
                 n, k,
                 c_one, dT, dT_offset,  lddt,
                        dwork, dwork_offset, ldwork, queue );

    // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
    magma_sgemm( notransV, MagmaTrans,
                 m , n, k,
                 c_neg_one, dV,    dV_offset,    lddv,
                            dwork, dwork_offset, ldwork,
                 c_one,     dC,    dC_offset,    lddc, queue );
    }
  else
    {
    // Form C H or C H^H
    // Comments assume C H.
    // When forming C H^H, T gets transposed via trans.

    // W = C V
    magma_sgemm( MagmaNoTrans, notransV,
                 m, k, n,
                 c_one,  dC,    dC_offset,    lddc,
                         dV,    dV_offset,    lddv,
                 c_zero, dwork, dwork_offset, ldwork, queue );

    // W = W T = C V T
    magma_strmm( MagmaRight, uplo, trans, MagmaNonUnit,
                 m, k,
                 c_one, dT,     dT_offset,    lddt,
                        dwork,  dwork_offset, ldwork, queue );

    // C = C - W V^H = C - C V T V^H = C (I - V T V^H) = C H
    magma_sgemm( MagmaNoTrans, transV,
                 m, n, k,
                 c_neg_one, dwork, dwork_offset, ldwork,
                            dV,    dV_offset,    lddv,
                 c_one,     dC,    dC_offset,    lddc, queue );
    }

  return info;
  } /* magma_slarfb */
