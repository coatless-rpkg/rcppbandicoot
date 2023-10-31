// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
COOT_FN(PREFIX,equ_array_div_scalar_pre)(__global eT2* dest,
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
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD src_index = row + col * src_M_n_rows + src_offset;
  const UWORD dest_index = row + col * dest_M_n_rows + dest_offset;

  if (row < n_rows && col < n_cols)
    {
    if (val_post == (eT2) (0))
      {
      // if both are 0, we take it as val_pre == 0 and val_post unused
      dest[dest_index] = (eT2) (val_pre / src[src_index]);
      }
    else if (val_pre == (eT1) (0) && val_post != (eT2) (0))
      {
      dest[dest_index] = val_post / ((eT2) src[src_index]);
      }
    else
      {
      // if both are nonzero, we apply sequentially---be careful!
      dest[dest_index] = val_post / ((eT2) (val_pre / src[src_index]));
      }
    }
  }
