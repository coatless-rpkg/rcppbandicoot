// Copyright 2026 Andrew Furey (http://andrew.industries)
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
COOT_FN(PREFIX,flipud)(__global eT1* out,
                       const UWORD out_offset,
                       __global const eT1* in,
                       const UWORD in_offset,
                       const UWORD n_rows,
                       const UWORD n_cols,
                       // subviews
                       const UWORD src_M_n_rows)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if( (row < n_rows) && (col < n_cols) )
    {
    const UWORD in_index = in_offset + col * src_M_n_rows + row;

    const UWORD out_row = n_rows - row - 1;
    const UWORD out_index = out_offset + col * n_rows + out_row;

    out[out_index] = in[in_index];
    }
  }
