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
COOT_FN(PREFIX,equ_array_minus_array_cube)(eT3* dest,
                                           const eT1* src_A,
                                           const eT2* src_B,
                                           // logical size of source and destination
                                           const UWORD n_rows,
                                           const UWORD n_cols,
                                           const UWORD n_slices,
                                           const UWORD dest_M_n_rows,
                                           const UWORD dest_M_n_cols,
                                           const UWORD src_A_M_n_rows,
                                           const UWORD src_A_M_n_cols,
                                           const UWORD src_B_M_n_rows,
                                           const UWORD src_B_M_n_cols)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD slice = blockIdx.z * blockDim.z + threadIdx.z;

  if (row < n_rows && col < n_cols && slice < n_slices)
    {
    const UWORD A_index = row + col * src_A_M_n_rows + slice * src_A_M_n_rows * src_A_M_n_cols;
    const UWORD B_index = row + col * src_B_M_n_rows + slice * src_B_M_n_rows * src_B_M_n_cols;
    const UWORD dest_index = row + col * dest_M_n_rows + slice * dest_M_n_rows * dest_M_n_cols;

    dest[dest_index] = TO_ET3(src_A[A_index] - src_B[B_index]);
    }
  }
