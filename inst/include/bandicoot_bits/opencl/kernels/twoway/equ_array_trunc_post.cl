// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,equ_array_trunc_post)(__global eT2* dest,
                                     const UWORD dest_offset,
                                     __global const eT1* src,
                                     const UWORD src_offset,
                                     const eT1 val_pre,
                                     const eT2 val_post,
                                     const UWORD n_rows,
                                     const UWORD n_cols,
                                     const UWORD dest_M_n_rows,
                                     const UWORD src_M_n_rows)
  {
  (void)(val_pre);
  (void)(val_post);

  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD src_index = row + col * src_M_n_rows + src_offset;
  const UWORD dest_index = row + col * dest_M_n_rows + dest_offset;

  if (row < n_rows && col < n_cols)
    {
    if (COOT_FN(coot_is_fp_,eT1)())
      {
      const fp_eT1 val = (fp_eT1) src[src_index];
      dest[dest_index] = (eT2) trunc(val);
      }
    else
      {
      dest[dest_index] = (eT2) src[src_index];
      }
    }
  }
