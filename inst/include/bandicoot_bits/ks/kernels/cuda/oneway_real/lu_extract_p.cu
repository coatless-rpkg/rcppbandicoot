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

__global__
void
COOT_FN(PREFIX,lu_extract_p)(eT1* P,
                             const UWORD* ipiv2,
                             const UWORD n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
    {
    const UWORD index = row + ipiv2[row] * n_rows;
    P[index] = (UWORD) 1;
    }
  }
