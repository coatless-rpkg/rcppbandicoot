// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
//~
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//~
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

__kernel
void
COOT_FN(PREFIX,inplace_xorwow_randu)(__global eT1* mem,
                                     __global uint_eT1* xorwow_state,
                                     const UWORD n)
  {
  const UWORD tid = get_global_id(0);
  const UWORD num_threads = get_global_size(0);
  UWORD i = tid;

  // Copy RNG state to local memory.
  uint_eT1 local_xorwow_state[6];
  local_xorwow_state[0] = xorwow_state[6 * tid    ];
  local_xorwow_state[1] = xorwow_state[6 * tid + 1];
  local_xorwow_state[2] = xorwow_state[6 * tid + 2];
  local_xorwow_state[3] = xorwow_state[6 * tid + 3];
  local_xorwow_state[4] = xorwow_state[6 * tid + 4];
  local_xorwow_state[5] = xorwow_state[6 * tid + 5];

  while (i < n)
    {
    uint_eT1 t = COOT_FN(xorwow_rng_,uint_eT1)(local_xorwow_state);
    // Now normalize to [0, 1] and compute the output.
    mem[i] = (eT1) (t / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());
    i += num_threads;
    }

  // Return updated RNG state to global memory.
  xorwow_state[6 * tid    ] = local_xorwow_state[0];
  xorwow_state[6 * tid + 1] = local_xorwow_state[1];
  xorwow_state[6 * tid + 2] = local_xorwow_state[2];
  xorwow_state[6 * tid + 3] = local_xorwow_state[3];
  xorwow_state[6 * tid + 4] = local_xorwow_state[4];
  xorwow_state[6 * tid + 5] = local_xorwow_state[5];
  }
