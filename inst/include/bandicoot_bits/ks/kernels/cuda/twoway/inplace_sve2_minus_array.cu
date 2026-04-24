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

__global__
void
COOT_FN(PREFIX,inplace_sve2_minus_array)(eT2* dest,
                                         const UWORD* dest_row_locs,
                                         const UWORD* dest_col_locs,
                                         const eT1* src,
                                         const UWORD n_rows,
                                         const UWORD n_cols,
                                         const UWORD dest_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n_rows && col < n_cols)
    {
    const UWORD dest_loc = ((dest_row_locs == NULL) ? row : dest_row_locs[row]) +
        (dest_n_rows * ((dest_col_locs == NULL) ? col : dest_col_locs[col]));
    const UWORD src_loc = row + col * n_rows;

    dest[dest_loc] -= TO_ET2(src[src_loc]);
    }
  }
