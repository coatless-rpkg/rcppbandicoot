// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,var_colwise)(__global eT1* dest,
                            const UWORD dest_offset,
                            __global const eT1* src,
                            const UWORD src_offset,
                            __global const eT1* src_means,
                            const UWORD src_means_offset,
                            const UWORD n_rows,
                            const UWORD n_cols,
                            const UWORD norm_correction,
                            const UWORD dest_mem_incr,
                            const UWORD src_M_n_rows,
                            const UWORD src_means_mem_incr)
  {
  const UWORD col = get_global_id(0);
  if(col < n_cols)
    {
    const __global eT1* colptr = &(src[src_offset + col * src_M_n_rows]);
    const eT1 mean_val = src_means[src_means_offset + col * src_means_mem_incr];
    eT1 acc = (eT1) (0);
    for (UWORD i = 0; i < n_rows; ++i)
      {
      eT1 val = (colptr[i] - mean_val);
      acc += (val * val);
      }

    dest[dest_offset + col * dest_mem_incr] = (acc / (eT1) (n_rows - norm_correction));
    }
  }