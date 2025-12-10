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
COOT_FN(PREFIX,fill)(eT1* out,
                     const eT1 val,
                     const UWORD n_rows,
                     const UWORD n_cols,
                     const UWORD M_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD index = col * M_n_rows + row;

  if(row < n_rows && col < n_cols)
    {
    out[index] = val;
    }
  }
