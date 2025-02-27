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

// multiply each column in `in` by the corresponding value in `A`
__kernel
void
COOT_FN(PREFIX,mul_colwise)(__global eT1* out,
                            const UWORD out_offset,
                            __global const eT1* A, // expected to have length n_cols
                            const UWORD A_offset,
                            __global const eT1* in,
                            const UWORD in_offset,
                            const eT1 alpha,
                            const UWORD n_rows,
                            const UWORD n_cols)
  {
  const UWORD col = get_global_id(0);
  if(col < n_cols)
    {
    const UWORD offset = col * n_rows;
    const eT1 val = alpha * A[A_offset + col];

    #pragma unroll
    for(UWORD i = 0; i < n_rows; ++i)
      {
      out[out_offset + i + offset] = val * in[in_offset + i + offset];
      }
    }
  }
