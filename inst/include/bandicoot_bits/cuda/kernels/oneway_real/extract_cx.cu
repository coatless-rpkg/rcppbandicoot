// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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

// Extract real or imaginary elements from a complex matrix into a real matrix.
// This kernel is a bit of a hack until we have actual complex matrix support!
__global__
void
COOT_FN(PREFIX,extract_cx)(const eT1* in_mem,
                           eT1* out_mem,
                           const UWORD real_or_imag,
                           const UWORD n_rows,
                           const UWORD n_cols,
                           const UWORD in_M_n_rows,
                           const UWORD out_M_n_rows)
  {
  // If real_or_imag is 0, we extract the real part.  If 1, we extract the
  // imaginary part.
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD in_index = 2 * (col * in_M_n_rows + row) + real_or_imag;
  const UWORD out_index = col * out_M_n_rows + row;

  if (col < n_cols && row < n_rows)
    {
    out_mem[out_index] = in_mem[in_index];
    }
  }
