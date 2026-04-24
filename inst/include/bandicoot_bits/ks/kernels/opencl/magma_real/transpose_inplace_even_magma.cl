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



// grid is ((n/nb) + 1) x (n/nb)/2, where n/nb is even.
// lower indicates blocks in strictly lower triangle of grid, excluding diagonal.
// lower blocks shift up by one to cover left side of matrix including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width
// to cover right side of matrix.
//      [ A00  A01 ]                  [ A10  .  |  .   .  ]
//      [ A10  A11 ]                  [ A20 A21 |  .   .  ]
// grid [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
//      [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
//      [ A40  A41 ]
//
// Each block is NB x NB threads.
// For non-diagonal block A, block B is symmetric block.
// Thread (i,j) loads A(i,j) into sA(j,i) and B(i,j) into sB(j,i), i.e., transposed,
// syncs, then saves sA(i,j) to B(i,j) and sB(i,j) to A(i,j).
// Threads outside the matrix do not touch memory.

__kernel
void
COOT_FN(PREFIX,transpose_inplace_even_magma)(const UWORD n,
                                             __global eT1* matrix,
                                             const UWORD matrix_offset,
                                             const UWORD lda)
  {
  matrix += matrix_offset;

  __local eT1 sA[MAGMABLAS_TRANS_INPLACE_NB][MAGMABLAS_TRANS_INPLACE_NB + 1];
  __local eT1 sB[MAGMABLAS_TRANS_INPLACE_NB][MAGMABLAS_TRANS_INPLACE_NB + 1];

  UWORD i = get_local_id(0);
  UWORD j = get_local_id(1);

  bool lower = (get_group_id(0) > get_group_id(1));
  UWORD ii = (lower ? (get_group_id(0) - 1) : (get_group_id(1) + get_num_groups(1)));
  UWORD jj = (lower ? (get_group_id(1)    ) : (get_group_id(0) + get_num_groups(1)));

  ii *= MAGMABLAS_TRANS_INPLACE_NB;
  jj *= MAGMABLAS_TRANS_INPLACE_NB;

  __global eT1* A = matrix + (ii + i) + (jj + j) * lda;
  if (ii == jj)
    {
    if (ii + i < n && jj + j < n)
      {
      sA[j][i] = *A;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ii + i < n && jj + j < n)
      {
      *A = sA[i][j];
      }
    }
  else
    {
    __global eT1* B = matrix + (jj + i) + (ii + j) * lda;
    if (ii + i < n && jj + j < n)
      {
      sA[j][i] = *A;
      }
    if (jj + i < n && ii + j < n)
      {
      sB[j][i] = *B;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ii + i < n && jj + j < n)
      {
      *A = sB[i][j];
      }
    if (jj + i < n && ii + j < n)
      {
      *B = sA[i][j];
      }
    }
  }
