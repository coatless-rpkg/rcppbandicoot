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
COOT_FN(PREFIX,equ_array_minus_scalar_pre_post)(__global eT2* dest,
                                                const UWORD dest_offset,
                                                __global const eT1* src,
                                                const UWORD src_offset,
                                                const eT1 val_pre,
                                                const eT2 val_post,
                                                const UWORD n_rows,
                                                const UWORD n_cols,
                                                const UWORD n_slices,
                                                const UWORD dest_M_n_rows,
                                                const UWORD dest_M_n_cols,
                                                const UWORD src_M_n_rows,
                                                const UWORD src_M_n_cols)
  {
  (void) val_post;

  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD slice = get_global_id(2);

  const UWORD src_index  = row + col * src_M_n_rows  + slice * src_M_n_rows * src_M_n_cols + src_offset;
  const UWORD dest_index = row + col * dest_M_n_rows + slice * dest_M_n_rows * dest_M_n_cols + dest_offset;

  if (row < n_rows && col < n_cols && slice < n_slices)
    {
    dest[dest_index] = (eT2) (val_pre - src[src_index]);
    }
  }
