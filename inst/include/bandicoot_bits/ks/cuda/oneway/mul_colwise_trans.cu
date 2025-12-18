// Copyright 2023-2025 Ryan Curtin (http://www.ratml.org/)
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

// multiply each column in `trans(in)` by the corresponding value in `A`
__global__
void
COOT_FN(PREFIX,mul_colwise_trans)(eT1* out,
                                  const eT1* A, // expected to have length n_cols
                                  const UWORD A_incr,
                                  const eT1* in,
                                  const eT1 alpha, // scalar to multiply
                                  const UWORD n_rows, // size of `out`
                                  const UWORD n_cols,
                                  const UWORD in_M_n_rows)
  {
  const UWORD col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col < n_cols)
    {
    const eT1 val = alpha * A[col];
    for (UWORD i = 0; i < n_rows; ++i)
      {
      const UWORD in_offset = col + i * in_M_n_rows;
      const UWORD out_offset = col * n_rows + i;
      out[out_offset] = val * in[in_offset];
      }
    }
  }
