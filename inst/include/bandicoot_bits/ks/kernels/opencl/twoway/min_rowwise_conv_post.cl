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
COOT_FN(PREFIX,min_rowwise_conv_post)(__global eT2* dest,
                                      const UWORD dest_offset,
                                      __global const eT1* src,
                                      const UWORD src_offset,
                                      const UWORD n_rows,
                                      const UWORD n_cols,
                                      const UWORD dest_mem_incr,
                                      const UWORD src_M_n_rows)
  {
  const UWORD row = get_global_id(0);
  if(row < n_rows)
    {
    eT1 acc = (eT1) src[src_offset + row];
    for(UWORD i = 1; i < n_cols; ++i)
      {
      acc = min(acc, src[src_offset + (i * src_M_n_rows) + row]);
      }
    dest[dest_offset + row * dest_mem_incr] = (eT2) (acc);
    }
  }
