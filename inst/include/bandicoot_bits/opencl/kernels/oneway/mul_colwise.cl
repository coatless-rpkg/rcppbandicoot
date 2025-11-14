// Copyright 2023-2025 Ryan Curtin (http://www.ratml.org)
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
                            const UWORD A_incr,
                            __global const eT1* in,
                            const UWORD in_offset,
                            const eT1 alpha,
                            const UWORD n_rows,
                            const UWORD n_cols,
                            const UWORD in_M_n_rows)
  {
  const UWORD col = get_global_id(0);
  if(col < n_cols)
    {
    const UWORD in_elem_offset  = col * in_M_n_rows;
    const UWORD out_elem_offset = col * n_rows;
    const eT1 val = alpha * A[A_offset + col * A_incr];

    #ifdef CL_VERSION_2_0
    __attribute__((opencl_unroll_hint))
    #endif
    for(UWORD i = 0; i < n_rows; ++i)
      {
      out[out_offset + i + out_elem_offset] = val * in[in_offset + i + in_elem_offset];
      }
    }
  }
