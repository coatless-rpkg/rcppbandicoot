// Copyright 2021 Marcus Edel (http://www.kurg.org/)
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

// philox_4x32_10, specific to generating u32s
// This depends on the functions in inplace_philox_4x32_10_rng.cl.

__kernel
void
COOT_FN(PREFIX,inplace_philox_randn)(__global eT1* mem,
                                     const UWORD mem_offset,
                                     __global uint* philox_state,
                                     const UWORD n,
                                     const fp_eT1 mu,
                                     const fp_eT1 sd)
  {
  const UWORD tid = get_global_id(0);
  const UWORD num_threads = get_global_size(0);
  UWORD i = tid;

  // Copy RNG state to local memory.
  uint local_philox_state[6];
  local_philox_state[0] = philox_state[6 * tid    ];
  local_philox_state[1] = philox_state[6 * tid + 1];
  local_philox_state[2] = philox_state[6 * tid + 2];
  local_philox_state[3] = philox_state[6 * tid + 3];
  local_philox_state[4] = philox_state[6 * tid + 4];
  local_philox_state[5] = philox_state[6 * tid + 5];

  // Only used if we are generating 64-bit types.
  uint aux_mem[4];

  while (i < n)
    {
    COOT_FN(philox_4x32_10_rng_,uint_eT1)(local_philox_state, aux_mem);

    // Perform the Box-Muller transformation to transform [0, 1] samples to N(0, 1).
    fp_eT1 sqrt_inner = -2 * log(COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 0) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());
    fp_eT1 trig_inner = 2 * M_PI * (COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 1) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());

    mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * cos(trig_inner)) * sd + mu);
    i += num_threads;
    if (i < n)
      mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * sin(trig_inner)) * sd + mu);
    i += num_threads;

    sqrt_inner = -2 * log(COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 2) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());
    trig_inner = 2 * M_PI * (COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 3) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());

    if (i < n)
      mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * cos(trig_inner)) * sd + mu);
    i += num_threads;
    if (i < n)
      mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * sin(trig_inner)) * sd + mu);
    }

  // Restore RNG state.
  philox_state[6 * tid    ] = local_philox_state[0];
  philox_state[6 * tid + 1] = local_philox_state[1];
  philox_state[6 * tid + 2] = local_philox_state[2];
  philox_state[6 * tid + 3] = local_philox_state[3];
  philox_state[6 * tid + 4] = local_philox_state[4];
  philox_state[6 * tid + 5] = local_philox_state[5];
  }
