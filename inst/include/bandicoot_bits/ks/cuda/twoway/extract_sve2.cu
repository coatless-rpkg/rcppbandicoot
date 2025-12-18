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



// this kernel is technically incorrect if the size is not a factor of 2!
__global__
void
COOT_FN(PREFIX,extract_sve2)(eT2* out_mem,
                             const eT1* in_mem,
                             const UWORD* in_row_locs,
                             const UWORD* in_col_locs,
                             const UWORD n_row_elems,
                             const UWORD n_col_elems,
                             const UWORD out_n_rows,
                             const UWORD in_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n_row_elems && col < n_col_elems)
    {
    const UWORD in_loc = ((in_row_locs == NULL) ? row : in_row_locs[row]) +
        in_n_rows * ((in_col_locs == NULL) ? col : in_col_locs[col]);

    out_mem[row + out_n_rows * col] = TO_ET2(in_mem[in_loc]);
    }
  }
