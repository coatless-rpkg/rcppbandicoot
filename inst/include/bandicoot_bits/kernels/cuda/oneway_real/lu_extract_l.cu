// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// This extracts L from U, and sets the lower diagonal of U to 0.
__global__
void
COOT_FN(PREFIX,lu_extract_l)(eT1* L,
                             eT1* U,
                             const eT1* in,
                             const UWORD n_rows,
                             const UWORD n_cols)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;

  // Note that neither U nor L must be square.
  //   L has size n_rows x min(n_rows, n_cols).
  //   U has size min(n_rows, n_cols) x n_cols.
  const UWORD min_rows_cols = min(n_rows, n_cols);

  const UWORD in_index = row + n_rows * col; // this is also L_out_index
  const UWORD U_out_index = row + min_rows_cols * col;

  if ((row < n_rows) && (col < min_rows_cols))
    {
    L[in_index] = (row > col) ? in[in_index] : ((row == col) ? 1 : 0);
    }

  if ((row < min_rows_cols) && (col < n_cols))
    {
    U[U_out_index] = (row > col) ? 0 : in[in_index];
    }
  }
