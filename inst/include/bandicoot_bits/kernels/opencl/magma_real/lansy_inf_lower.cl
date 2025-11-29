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



// Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
// where n is any size and A is stored lower.
// Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each (inf_bs=32).
// z precision uses > 16 KB shared memory, so requires Fermi (arch >= 200).



__kernel
void
COOT_FN(PREFIX,lansy_inf_lower)
  (
  const UWORD n,
  const __global eT1* A,
  const UWORD A_offset,
  const UWORD lda,
  __global eT1* dwork,
  const UWORD dwork_offset,
  const UWORD n_full_block,
  const UWORD n_mod_bs
  )
  {
  A += A_offset;
  dwork += dwork_offset;

  UWORD tx = get_local_id(0);
  UWORD ty = get_local_id(1);

  UWORD diag = get_group_id(0) * MAGMABLAS_LANSY_INF_BS;
  UWORD ind  = get_group_id(0) * MAGMABLAS_LANSY_INF_BS + tx;

  eT1 res = 0.;

  __local eT1 la[MAGMABLAS_LANSY_INF_BS][MAGMABLAS_LANSY_INF_BS + 1];

  if ( get_group_id(0) < n_full_block )
    {
    // ------------------------------
    // All full block rows
    A += ind;
    A += ty * lda;

    // ----------
    // loop over all blocks left of the diagonal block
    for(UWORD i=0; i < diag; i += MAGMABLAS_LANSY_INF_BS )
      {
      // 32x4 threads cooperatively load 32x32 block
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=0; j < MAGMABLAS_LANSY_INF_BS; j += 4)
        {
        la[tx][ty+j] = A[j*lda];
        }
      A += lda * MAGMABLAS_LANSY_INF_BS;
      barrier( CLK_LOCAL_MEM_FENCE );

      // compute 4 partial sums of each row, i.e.,
      // for ty=0:  res = sum( la[tx, 0: 7] )
      // for ty=1:  res = sum( la[tx, 8:15] )
      // for ty=2:  res = sum( la[tx,16:23] )
      // for ty=3:  res = sum( la[tx,24:31] )
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=ty*8; j < ty*8 + 8; j++)
        {
        res += ET1_ABS( la[tx][j] );
        }
      barrier( CLK_LOCAL_MEM_FENCE );
      }

    // ----------
    // load diagonal block
    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint(8)))
    #endif
    for(UWORD j=0; j < MAGMABLAS_LANSY_INF_BS; j += 4)
      {
      la[tx][ty+j] = A[j*lda];
      }
    barrier( CLK_LOCAL_MEM_FENCE );

    // copy lower triangle to upper triangle, and
    // make diagonal real (zero imaginary part)
    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint(8)))
    #endif
    for(UWORD i=ty*8; i < ty*8 + 8; i++)
      {
      if ( i < tx )
        {
        la[i][tx] = la[tx][i];
        }
      }
    barrier( CLK_LOCAL_MEM_FENCE );

    // partial row sums
    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint(8)))
    #endif
    for(UWORD j=ty*8; j < ty*8 + 8; j++)
      {
      res += ET1_ABS( la[tx][j] );
      }
    barrier( CLK_LOCAL_MEM_FENCE );

    // ----------
    // loop over all 32x32 blocks below diagonal block
    A += MAGMABLAS_LANSY_INF_BS;
    for(UWORD i=diag + MAGMABLAS_LANSY_INF_BS; i < n - n_mod_bs; i += MAGMABLAS_LANSY_INF_BS )
      {
      // load block (transposed)
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=0; j < MAGMABLAS_LANSY_INF_BS; j += 4)
        {
        la[ty+j][tx] = A[j*lda];
        }
      A += MAGMABLAS_LANSY_INF_BS;
      barrier( CLK_LOCAL_MEM_FENCE );

      // partial row sums
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=ty*8; j < ty*8 + 8; j++)
        {
        res += ET1_ABS( la[tx][j] );
        }
      barrier( CLK_LOCAL_MEM_FENCE );
      }

    // ----------
    // last partial block, which is (n_mod_bs by inf_bs)
    if ( n_mod_bs > 0 )
      {
      // load block (transposed), with zeros for rows outside matrix
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=0; j < MAGMABLAS_LANSY_INF_BS; j += 4)
        {
        if ( tx < n_mod_bs )
          {
          la[ty+j][tx] = A[j*lda];
          }
        else
          {
          la[ty+j][tx] = (eT1) 0;
          }
        }
      barrier( CLK_LOCAL_MEM_FENCE );

      // partial row sums
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=ty*8; j < ty*8 + 8; j++)
        {
        res += ET1_ABS( la[tx][j] );
        }
      barrier( CLK_LOCAL_MEM_FENCE );
      }

    // ----------
    // 32x4 threads store partial sums into shared memory
    la[tx][ty] = res;
    barrier( CLK_LOCAL_MEM_FENCE );

    // first column of 32x1 threads computes final sum of each row
    if ( ty == 0 )
      {
      res = res + la[tx][1] + la[tx][2] + la[tx][3];
      dwork[ind] = res;
      }
    }
  else
    {
    // ------------------------------
    // Last, partial block row
    // Threads past end of matrix (i.e., ind >= n) are redundantly assigned
    // the last row (n-1). At the end, those results are ignored -- only
    // results for ind < n are saved into dwork.
    if ( tx < n_mod_bs )
      {
      A += ind;
      }
    else
      {
      A += (get_group_id(0) * MAGMABLAS_LANSY_INF_BS + n_mod_bs - 1);  // redundantly do last row
      }
    A += ty * lda;

    // ----------
    // loop over all blocks left of the diagonal block
    // each is (n_mod_bs by inf_bs)
    for(UWORD i=0; i < diag; i += MAGMABLAS_LANSY_INF_BS )
      {
      // load block
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=0; j < MAGMABLAS_LANSY_INF_BS; j += 4)
        {
        la[tx][ty+j] = A[j*lda];
        }
      A += lda * MAGMABLAS_LANSY_INF_BS;
      barrier( CLK_LOCAL_MEM_FENCE );

      // partial row sums
      #ifdef CL_VERSION_2_0
      __attribute__((opencl_unroll_hint(8)))
      #endif
      for(UWORD j=0; j < 8; j++)
        {
        res += ET1_ABS( la[tx][j+ty*8] );
        }
      barrier( CLK_LOCAL_MEM_FENCE );
      }

    // ----------
    // partial diagonal block
    if ( ty == 0 && tx < n_mod_bs )
      {
      // sum rows left of diagonal
      for(UWORD j=0; j < tx; j++)
        {
        res += ET1_ABS( *A );
        A += lda;
        }
      // sum diagonal (ignoring imaginary part)
      res += ET1_ABS( *A );
      A += 1;
      // sum column below diagonal
      for(UWORD j=tx+1; j < n_mod_bs; j++)
        {
        res += ET1_ABS( *A );
        A += 1;
        }
      }
    barrier( CLK_LOCAL_MEM_FENCE );

    // ----------
    // 32x4 threads store partial sums into shared memory
    la[tx][ty]= res;
    barrier( CLK_LOCAL_MEM_FENCE );

    // first column of 32x1 threads computes final sum of each row
    // rows outside matrix are ignored
    if ( ty == 0 && tx < n_mod_bs )
      {
      res = res + la[tx][1] + la[tx][2] + la[tx][3];
      dwork[ind] = res;
      }
    }
  }
