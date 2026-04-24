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

__kernel
void
COOT_FN(PREFIX,min_colwise_conv_pre)(__global eT2* dest,
                                     const UWORD dest_offset,
                                     __global const eT1* src,
                                     const UWORD src_offset,
                                     const UWORD n_rows,
                                     const UWORD n_cols,
                                     const UWORD dest_mem_incr,
                                     const UWORD src_M_n_rows)
  {
  const UWORD col = get_global_id(0);
  if(col < n_cols)
    {
    __global const eT1* colptr = &(src[src_offset + col * src_M_n_rows]);
    eT2 acc = (eT2) colptr[0];
    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint))
    #endif
    for(UWORD i = 1; i < n_rows; ++i)
      {
      acc = min(acc, (eT2) (colptr[i]));
      }
    dest[dest_offset + col * dest_mem_incr] = acc;
    }
  }
