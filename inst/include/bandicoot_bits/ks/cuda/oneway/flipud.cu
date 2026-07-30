// Copyright Andrew Furey (http://andrew.industries)
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
COOT_FN(PREFIX,flipud)(eT1* out,
                       const eT1* in,
                       const UWORD n_rows,
                       const UWORD n_cols,
                       // subviews
                       const UWORD src_M_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;

  if(row < n_rows && col < n_cols)
    {
    const UWORD in_index = row + col * src_M_n_rows;

    const UWORD out_row = n_rows - row - 1;
    const UWORD out_index = out_row + col * n_rows;

    out[out_index] = in[in_index];
    }
  }
