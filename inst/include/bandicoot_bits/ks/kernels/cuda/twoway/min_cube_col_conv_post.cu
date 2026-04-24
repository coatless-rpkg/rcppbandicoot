// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,min_cube_col_conv_post)(eT2* dest,
                                       const eT1* src,
                                       const UWORD n_rows,
                                       const UWORD n_cols,
                                       const UWORD n_slices)
  {
  const UWORD row   = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD slice = blockIdx.y * blockDim.y + threadIdx.y;

  if(row < n_rows && slice < n_slices)
    {
    eT1 acc = src[row + slice * n_rows * n_cols];
    for (UWORD i = 1; i < n_cols; ++i)
      {
      acc = min(acc, src[(i * n_rows) + row + slice * n_rows * n_cols]);
      }

    dest[row + slice * n_rows] = TO_ET2(acc);
    }
  }
