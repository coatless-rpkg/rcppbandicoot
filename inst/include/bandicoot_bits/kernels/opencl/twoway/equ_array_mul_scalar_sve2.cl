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
COOT_FN(PREFIX,equ_array_mul_scalar_sve2)(__global eT2* dest,
                                          const UWORD dest_offset,
                                          __global const UWORD* dest_row_locs,
                                          const UWORD dest_row_locs_offset,
                                          __global const UWORD* dest_col_locs,
                                          const UWORD dest_col_locs_offset,
                                          __global const eT1* src,
                                          const UWORD src_offset,
                                          __global const UWORD* src_row_locs,
                                          const UWORD src_row_locs_offset,
                                          __global const UWORD* src_col_locs,
                                          const UWORD src_col_locs_offset,
                                          const eT1 val_pre,
                                          const eT2 val_post,
                                          const UWORD n_row_elems,
                                          const UWORD n_col_elems,
                                          const UWORD dest_n_rows,
                                          const UWORD src_n_rows)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if (row < n_row_elems && col < n_col_elems)
    {
    const UWORD src_loc = src_offset +
        ((src_row_locs == NULL) ? row : src_row_locs[row + src_row_locs_offset]) +
        src_n_rows * ((src_col_locs == NULL) ? col : src_col_locs[col + src_col_locs_offset]);

    const UWORD dest_loc = dest_offset +
        ((dest_row_locs == NULL) ? row : dest_row_locs[row + dest_row_locs_offset]) +
        dest_n_rows * ((dest_col_locs == NULL) ? col : dest_col_locs[col + dest_col_locs_offset]);

    dest[dest_loc] = ((eT2) (src[src_loc] * val_pre)) * val_post;
    }
  }
