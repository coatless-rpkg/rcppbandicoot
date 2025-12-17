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



// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
// for each subtile
//     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
//     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
//     A  += NX
//     AT += NX*ldat

__kernel
void
COOT_FN(PREFIX,transpose_magma)(const UWORD m,
                                const UWORD n,
                                __global const eT1* A,
                                const UWORD A_offset,
                                const UWORD lda,
                                __global eT1* AT,
                                const UWORD AT_offset,
                                const UWORD ldat)
  {
  A += A_offset;
  AT += AT_offset;

  __local eT1 sA[MAGMABLAS_TRANS_NB][MAGMABLAS_TRANS_NX+1];

  UWORD tx  = get_local_id(0);
  UWORD ty  = get_local_id(1);
  UWORD ibx = get_group_id(0) * MAGMABLAS_TRANS_NB;
  UWORD iby = get_group_id(1) * MAGMABLAS_TRANS_NB;
  UWORD i, j;

  A  += ibx + tx + (iby + ty) * lda;
  AT += iby + tx + (ibx + ty) * ldat;

  #ifdef CL_VERSION_2_0
  __attribute__((opencl_unroll_hint))
  #endif
  for (int tile = 0; tile < MAGMABLAS_TRANS_NB / MAGMABLAS_TRANS_NX; ++tile)
    {
    // load NX-by-NB subtile transposed from A into sA
    i = ibx + tx + tile * MAGMABLAS_TRANS_NX;
    j = iby + ty;
    if (i < m)
      {
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint))
      #endif
      for (int j2=0; j2 < MAGMABLAS_TRANS_NB; j2 += MAGMABLAS_TRANS_NY)
        {
        if (j + j2 < n)
          {
          sA[ty + j2][tx] = A[j2*lda];
          }
        }
      }

    barrier(CLK_LOCAL_MEM_FENCE);

    // save NB-by-NX subtile from sA into AT
    i = iby + tx;
    j = ibx + ty + tile * MAGMABLAS_TRANS_NX;
    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint))
    #endif
    for (int i2 = 0; i2 < MAGMABLAS_TRANS_NB; i2 += MAGMABLAS_TRANS_NX)
      {
      if (i + i2 < n)
        {
        #ifdef CL_VERSION_2_0
        __attribute__((opencl_unroll_hint))
        #endif
        for (int j2 = 0; j2 < MAGMABLAS_TRANS_NX; j2 += MAGMABLAS_TRANS_NY)
          {
          if (j + j2 < m)
            {
            AT[i2 + j2 * ldat] = sA[tx + i2][ty + j2];
            }
          }
        }
      }

    barrier(CLK_LOCAL_MEM_FENCE);

    // move to next subtile
    A  += MAGMABLAS_TRANS_NX;
    AT += MAGMABLAS_TRANS_NX * ldat;
    }
  }
