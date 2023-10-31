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



// Put 0s in the upper triangular part of a panel and 1s on the diagonal.
// Stores previous values in work array, to be restored later with magma_dq_to_panel().

inline
void
magma_dpanel_to_q(magma_uplo_t uplo, magma_int_t ib, double *A, magma_int_t lda, double *work)
  {
  magma_int_t i, j, k = 0;
  double *col;
  double c_zero = MAGMA_D_ZERO;
  double c_one  = MAGMA_D_ONE;

  if (uplo == MagmaUpper)
    {
    for (i = 0; i < ib; ++i)
      {
      col = A + i*lda;
      for (j = 0; j < i; ++j)
        {
        work[k] = col[j];
        col [j] = c_zero;
        ++k;
        }

        work[k] = col[i];
        col [j] = c_one;
        ++k;
        }
    }
  else
    {
    for (i=0; i < ib; ++i)
      {
      col = A + i*lda;
      work[k] = col[i];
      col [i] = c_one;
      ++k;
      for (j=i+1; j < ib; ++j)
        {
        work[k] = col[j];
        col [j] = c_zero;
        ++k;
        }
      }
    }
  }



// Restores a panel, after call to magma_dpanel_to_q().

inline
void
magma_dq_to_panel(magma_uplo_t uplo, magma_int_t ib, double *A, magma_int_t lda, double *work)
  {
  magma_int_t i, j, k = 0;
  double *col;

  if (uplo == MagmaUpper)
    {
    for (i = 0; i < ib; ++i)
      {
      col = A + i*lda;
      for (j = 0; j <= i; ++j)
        {
        col[j] = work[k];
        ++k;
        }
      }
    }
  else
    {
    for (i = 0; i < ib; ++i)
      {
      col = A + i*lda;
      for (j = i; j < ib; ++j)
        {
        col[j] = work[k];
        ++k;
        }
      }
    }
  }
