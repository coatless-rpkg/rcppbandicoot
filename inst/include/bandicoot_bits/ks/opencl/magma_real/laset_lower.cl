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
// clMAGMA 1.3 (2014-11-14).
// clMAGMA 1.3 is distributed under a 3-clause BSD license as follows:
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



// Similar to laset_full, but updates only the diagonal and below.
// Blocks that are fully above the diagonal exit immediately.
//
// Code similar to lacpy, zlat2c, clat2z.

__kernel
void
COOT_FN(PREFIX,laset_lower)(const UWORD m,
                            const UWORD n,
                            const eT1 offdiag,
                            const eT1 diag,
                            __global eT1* A,
                            const UWORD A_offset,
                            const UWORD lda)
  {
  A += A_offset;

  UWORD ind = get_group_id(0) * MAGMABLAS_BLK_X + get_local_id(0);
  UWORD iby = get_group_id(1) * MAGMABLAS_BLK_Y;
  /* check if full block-column && (below diag) */
  bool full = (iby + MAGMABLAS_BLK_Y <= n && (ind >= iby + MAGMABLAS_BLK_Y));
  /* do only rows inside matrix, and blocks not above diag */
  if (ind < m && ind + MAGMABLAS_BLK_X > iby)
    {
    A += ind + iby*lda;
    if (full)
      {
      // full block-column, off-diagonal block
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint))
      #endif
      for (int j=0; j < MAGMABLAS_BLK_Y; ++j)
        {
        A[j * lda] = offdiag;
        }
      }
    else
      {
      // either partial block-column or diagonal block
      for (int j=0; j < MAGMABLAS_BLK_Y && iby+j < n; ++j)
        {
        if (iby + j == ind)
          A[j * lda] = diag;
        else if (ind > iby + j)
          A[j*lda] = offdiag;
        }
      }
    }
  }
