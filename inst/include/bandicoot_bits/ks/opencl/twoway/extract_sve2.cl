// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2025 Ryan Curtin (http://ratml.org)
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
COOT_FN(PREFIX,extract_sve2)(__global eT2* out_mem,
                             const UWORD out_mem_offset,
                             __global const eT1* in_mem,
                             const UWORD in_mem_offset,
                             __global const UWORD* in_row_locs,
                             const UWORD in_row_locs_offset,
                             __global const UWORD* in_col_locs,
                             const UWORD in_col_locs_offset,
                             const UWORD n_row_elems,
                             const UWORD n_col_elems,
                             const UWORD out_n_rows,
                             const UWORD in_n_rows)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if (row < n_row_elems && col < n_col_elems)
    {
    const UWORD in_loc = in_mem_offset +
        ((in_row_locs == NULL) ? row : in_row_locs[row + in_row_locs_offset]) +
        in_n_rows * ((in_col_locs == NULL) ? col : in_col_locs[col + in_col_locs_offset]);

    const UWORD out_loc = out_mem_offset + row + out_n_rows * col;

    out_mem[out_loc] = (eT2) in_mem[in_loc];
    }
  }
