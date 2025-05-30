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
COOT_FN(PREFIX,rel_all_neq_rowwise)(__global UWORD* out,
                                    const UWORD out_offset,
                                    __global const eT1* A,
                                    const UWORD A_offset,
                                    const eT2 val,
                                    const UWORD A_n_rows,
                                    const UWORD A_n_cols)
  {
  const UWORD row = get_global_id(0);
  if(row < A_n_rows)
    {
    UWORD result = 1;
    for(UWORD i = 0; i < A_n_cols; ++i)
      {
      const eT2 val1 = (eT2) A[i * A_n_rows + row + A_offset];
      result &= (val1 != val);
      }
    out[row + out_offset] = result;
    }
  }
