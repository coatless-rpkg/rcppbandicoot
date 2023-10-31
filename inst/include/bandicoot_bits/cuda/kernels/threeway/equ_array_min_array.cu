// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,equ_array_min_array)(eT3* dest,
                                    const eT1* src_A,
                                    const eT2* src_B,
                                    // logical size of source and destination
                                    const UWORD n_rows,
                                    const UWORD n_cols,
                                    const UWORD dest_M_n_rows,
                                    const UWORD src_A_M_n_rows,
                                    const UWORD src_B_M_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n_rows && col < n_cols)
    {
    const UWORD A_index = row + col * src_A_M_n_rows;
    const UWORD B_index = row + col * src_B_M_n_rows;
    const UWORD dest_index = row + col * dest_M_n_rows;

    const threeway_promoted_eT a_val = (threeway_promoted_eT) src_A[A_index];
    const threeway_promoted_eT b_val = (threeway_promoted_eT) src_B[B_index];
    dest[dest_index] = (eT3) min(a_val, b_val);
    }
  }
