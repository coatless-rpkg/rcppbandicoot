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
COOT_FN(PREFIX,cross)(__global eT2* out,
                      const UWORD out_offset,
                      __global const eT1* A,
                      const UWORD A_offset,
                      __global const eT1* B,
                      const UWORD B_offset) // A and B should have 3 elements
  {
  const UWORD idx = get_global_id(0);

  if (idx < 3)
    {
    const UWORD a1_index = ((idx + 1) % 3) + A_offset;
    const UWORD a2_index = ((idx + 2) % 3) + A_offset;

    const UWORD b1_index = ((idx + 2) % 3) + B_offset;
    const UWORD b2_index = ((idx + 1) % 3) + B_offset;

    const eT1 val = (A[a1_index] * B[b1_index]) - (A[a2_index] * B[b2_index]);
    out[idx + out_offset] = (eT2) val;
    }
  }
