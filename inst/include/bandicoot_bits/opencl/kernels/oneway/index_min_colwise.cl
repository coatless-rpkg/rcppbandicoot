// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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
COOT_FN(PREFIX,index_min_colwise)(__global UWORD* dest,
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

    eT1 best_val = colptr[0];
    UWORD best_index = 0;
    #pragma unroll
    for(UWORD i = 1; i < n_rows; ++i)
      {
      if (colptr[i] < best_val)
        {
        best_val = colptr[i];
        best_index = i;
        }
      }
    dest[dest_offset + col * dest_mem_incr] = best_index;
    }
  }
