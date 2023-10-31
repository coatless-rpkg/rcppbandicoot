// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



__kernel
void
COOT_FN(PREFIX,symmatl)(__global eT2* out,
                        __global const eT1* A,
                        const UWORD size)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  if (row < size && col < size && row >= col)
    {
    const eT2 val = (eT2) A[row + size * col];

    out[row + size * col] = val;
    out[col + size * row] = val;
    }
  }
