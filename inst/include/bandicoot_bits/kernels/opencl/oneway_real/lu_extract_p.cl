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

__kernel
void
COOT_FN(PREFIX,lu_extract_p)(__global eT1* P,
                             const UWORD P_offset,
                             __global const UWORD* ipiv2,
                             const UWORD n_rows)
  {
  const UWORD row = get_global_id(0);

  if (row < n_rows)
    {
    const UWORD index = row + ipiv2[row] * n_rows;
    P[P_offset + index] = (eT1) 1;
    }
  }
