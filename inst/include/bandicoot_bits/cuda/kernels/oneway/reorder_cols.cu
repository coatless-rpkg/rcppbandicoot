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

__global__
void
COOT_FN(PREFIX,reorder_cols)(eT1* out_mem,
                             const eT1* in_mem,
                             const UWORD n_rows,
                             const UWORD* ordering,
                             const UWORD out_n_cols)
  {
  const UWORD out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_col < out_n_cols)
    {
    const UWORD in_col = ordering[out_col];

          eT1* out_colptr = out_mem + (out_col * n_rows);
    const eT1* in_colptr  = in_mem  + (in_col * n_rows);

    for (UWORD i = 0; i < n_rows; ++i)
      {
      out_colptr[i] = in_colptr[i];
      }
    }
  }
