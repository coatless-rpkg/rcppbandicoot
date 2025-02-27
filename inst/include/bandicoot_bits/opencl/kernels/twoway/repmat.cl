// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,repmat)(__global const eT1* in,
                       const UWORD in_offset,
                       __global eT2* out,
                       const UWORD out_offset,
                       const UWORD n_rows,
                       const UWORD n_cols,
                       const UWORD copies_per_row,
                       const UWORD copies_per_col,
                       const UWORD new_n_rows)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD offset = row + col * n_rows;
  const eT2 element = (eT2) in[in_offset + offset];
  if( (row < n_rows) && (col < n_cols) )
    {
    for (UWORD c_copy = 0; c_copy < copies_per_col; ++c_copy)
      {
      const UWORD col_offset = (col + n_cols * c_copy) * new_n_rows;
      for (UWORD r_copy = 0; r_copy < copies_per_row; ++r_copy)
        {
        const UWORD copy_offset = col_offset + (row + n_rows * r_copy);
        out[out_offset + copy_offset] = element;
        }
      }
    }
  }
