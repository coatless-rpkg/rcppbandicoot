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
COOT_FN(PREFIX,inplace_xorwow_randi)(__global eT1* mem,
                                     const UWORD mem_offset,
                                     __global uint_eT1* xorwow_state,
                                     const UWORD n,
                                     const eT1 lo,
                                     const uint_eT1 range,
                                     const char needs_modulo)
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
    // This generates a number in [0, uint_eT1_max].
    uint_eT1 t = COOT_FN(xorwow_rng_,uint_eT1)(local_xorwow_state);
    // Modulo down to the range [0, (hi - lo)], if needed.
    if (needs_modulo == 1)
      t %= (range + 1);
    // Cast back to the correct type, and add lo to get the correct range.
    mem[mem_offset + i] = ((eT1) t) + lo;
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
