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



// See algorithm "xorwow" from page 5 of "Xorshift RNGs" by George Marsaglia.
inline
uint
xorwow32_rng_uint(uint* xorwow_state)
  {
  // xorwow_state[0] through xorwow_state[4] represent the 5 state integers,
  // and xorwow_state[5] holds the counter.
  uint t = xorwow_state[4] ^ (xorwow_state[4] >> 2);

  xorwow_state[4] = xorwow_state[3];
  xorwow_state[3] = xorwow_state[2];
  xorwow_state[2] = xorwow_state[1];
  xorwow_state[1] = xorwow_state[0];
  xorwow_state[0] ^= (xorwow_state[0] << 4) ^ (t ^ (t << 1));

  // Following Saito and Matsumoto (2012), we use a larger constant for d so that the higher bits flip more often.
  // We ignore their conclusion that XORWOW has problems (it's fast!).
  xorwow_state[5] += 268183997;
  return xorwow_state[0] + xorwow_state[5];
  }



__kernel
void
COOT_FN(PREFIX,inplace_xorwow32_randu)(__global eT1* mem,
                                       const UWORD mem_offset,
                                       __global uint* xorwow_state,
                                       const UWORD n)
  {
  const UWORD tid = get_global_id(0);
  const UWORD num_threads = get_global_size(0);
  UWORD i = tid;

  // Copy RNG state to local memory.
  uint local_xorwow_state[6];
  local_xorwow_state[0] = xorwow_state[6 * tid    ];
  local_xorwow_state[1] = xorwow_state[6 * tid + 1];
  local_xorwow_state[2] = xorwow_state[6 * tid + 2];
  local_xorwow_state[3] = xorwow_state[6 * tid + 3];
  local_xorwow_state[4] = xorwow_state[6 * tid + 4];
  local_xorwow_state[5] = xorwow_state[6 * tid + 5];

  while (i < n)
    {
    uint_eT1 t = (uint_eT1) xorwow32_rng_uint(local_xorwow_state);
    // Now normalize to [0, 1] and compute the output.
    mem[mem_offset + i] = (eT1) (t / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());
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
