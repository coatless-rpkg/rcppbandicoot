// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,strans)(eT2* out,
                       const eT1* in,
                       const UWORD in_n_rows,
                       const UWORD in_n_cols)
  {
  // For a non-inplace transpose, we can use a pretty naive approach.
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD in_offset = row + col * in_n_rows;
  const UWORD out_offset = col + row * in_n_cols;

  if( (row < in_n_rows) && (col < in_n_cols) )
    {
    const eT2 element = (eT2) in[in_offset];
    out[out_offset] = element;
    }
  }
