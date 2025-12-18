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
COOT_FN(PREFIX,fill_sve2)(__global eT1* out,
                          const UWORD out_offset,
                          __global const UWORD* out_row_locs,
                          const UWORD out_row_locs_offset,
                          __global const UWORD* out_col_locs,
                          const UWORD out_col_locs_offset,
                          const eT1 val,
                          const UWORD n_row_elems,
                          const UWORD n_col_elems,
                          const UWORD out_n_rows)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if (row < n_row_elems && col < n_col_elems)
    {
    const UWORD out_loc = out_offset +
        ((out_row_locs == NULL) ? row : out_row_locs[row + out_row_locs_offset]) +
        out_n_rows * ((out_col_locs == NULL) ? col : out_col_locs[col + out_col_locs_offset]);

    out[out_loc] = val;
    }
  }
