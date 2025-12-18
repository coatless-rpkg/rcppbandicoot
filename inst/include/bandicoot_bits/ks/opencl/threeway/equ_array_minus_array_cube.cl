// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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
COOT_FN(PREFIX,equ_array_minus_array_cube)(__global eT3* dest,
                                           const UWORD dest_offset,
                                           __global const eT1* src_A,
                                           const UWORD src_A_offset,
                                           __global const eT2* src_B,
                                           const UWORD src_B_offset,
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
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD slice = get_global_id(2);

  if (row < n_rows && col < n_cols && slice < n_slices)
    {
    const UWORD src_A_index = row + col * src_A_M_n_rows + src_A_offset + slice * src_A_M_n_rows * src_A_M_n_cols;
    const UWORD src_B_index = row + col * src_B_M_n_rows + src_B_offset + slice * src_B_M_n_rows * src_B_M_n_cols;
    const UWORD  dest_index = row + col *  dest_M_n_rows +  dest_offset + slice * dest_M_n_rows * dest_M_n_cols;

    dest[dest_index] = (eT3) (src_A[src_A_index] - src_B[src_B_index]);
    }
  }
