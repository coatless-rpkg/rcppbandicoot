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
COOT_FN(PREFIX,rotate_180)(__global eT1* out,
                           __global const eT1* in,
                           const UWORD n_rows,
                           const UWORD n_cols)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if( (row < n_rows) && (col < n_cols) )
    {
    const UWORD in_index = col * n_rows + row;
    // out(i, j) = in(n_rows - i - 1, n_cols - j - 1)
    //    or
    // out(n_rows - i - 1, n_cols - j - 1) = in(i, j)
    const UWORD out_row = n_rows - row - 1;
    const UWORD out_col = n_cols - col - 1;
    const UWORD out_index = out_col * n_rows + out_row;

    out[out_index] = in[in_index];
    }
  }
