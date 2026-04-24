// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,broadcast_plus)(__global eT2* out,
                               const UWORD out_offset,
                               __global const eT2* out_src,
                               const UWORD out_src_offset,
                               __global const eT1* in,
                               const UWORD in_offset,
                               const UWORD n_rows,
                               const UWORD n_cols,
                               const UWORD copies_per_row,
                               const UWORD copies_per_col,
                               const UWORD out_M_n_rows,
                               const UWORD out_src_M_n_rows,
                               const UWORD in_M_n_rows)
  {
  const UWORD out_row = get_global_id(0);
  const UWORD out_col = get_global_id(1);

  if ((out_row < n_rows * copies_per_row) && (out_col < n_cols * copies_per_col))
    {
    const UWORD in_row = out_row % n_rows;
    const UWORD in_col = out_col % n_cols;

    const UWORD out_loc     = out_col *     out_M_n_rows + out_row;
    const UWORD out_src_loc = out_col * out_src_M_n_rows + out_row;
    const UWORD  in_loc     =  in_col *      in_M_n_rows +  in_row;

    out[out_offset + out_loc] = out_src[out_src_offset + out_src_loc] + ((eT2) in[in_offset + in_loc]);
    }
  }
