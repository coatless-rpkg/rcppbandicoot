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



inline
void
philox_4x32_10_single_round(uint* counter, uint* key)
  {
  uint hi0 = mul_hi((uint) 0xD2511F53, counter[0]);
  uint hi1 = mul_hi((uint) 0xCD9E8D57, counter[2]);
  uint lo0 = 0xD2511F53 * counter[0];
  uint lo1 = 0xCD9E8D57 * counter[2];

  counter[0] = hi1 ^ counter[1] ^ key[0];
  counter[1] = lo1;
  counter[2] = hi0 ^ counter[3] ^ key[1];
  counter[3] = lo0;
  }



inline
void
philox_4x32_10_p_step(uint* philox_state)
  {
  if (++philox_state[0])
    return;
  if (++philox_state[1])
    return;
  if (++philox_state[2])
    return;
  ++philox_state[3];
  }



inline
void
philox_4x32_10_rng(uint* philox_state)
  {
  // 4 uint counter:     philox_state[0:3]
  // 2 uint key:         philox_state[4:5]

  // apply P (increment state)
  philox_4x32_10_p_step(philox_state);

  // apply S-box 10 times
  for (UWORD i = 0; i < 9; ++i)
    {
    philox_4x32_10_single_round(philox_state, philox_state + 4);
    philox_state[4] += 0x9E3779B9;
    philox_state[5] += 0xBB67AE85;
    }
  philox_4x32_10_single_round(philox_state, philox_state + 4);
  }


//
// Convenience functions to get random numbers of 32-bit or 64-bit width out of the Philox 4x32-10 generator.
// The aux memory is only used for ulongs (64-bit width).
//

inline
void
philox_4x32_10_rng_uchar(uint* philox_state, uint* aux)
  {
  // This generates more than we need, but that's okay.
  philox_4x32_10_rng(philox_state);
  }



inline
void
philox_4x32_10_rng_ushort(uint* philox_state, uint* aux)
  {
  // This generates more than we need, but that's okay.
  philox_4x32_10_rng(philox_state);
  }



inline
void
philox_4x32_10_rng_uint(uint* philox_state, uint* aux)
  {
  philox_4x32_10_rng(philox_state);
  }



inline
void
philox_4x32_10_rng_ulong(uint* philox_state, uint* aux)
  {
  philox_4x32_10_rng(philox_state);
  // Save 4x32 bits of random data.
  aux[0] = philox_state[0];
  aux[1] = philox_state[1];
  aux[2] = philox_state[2];
  aux[3] = philox_state[3];
  // Generate the next 4x32 bits of random data.
  philox_4x32_10_rng(philox_state);
  }



inline
uchar
philox_get_elem_uchar(uint* philox_state, uint* aux, const UWORD i)
  {
  return ((uchar*) philox_state)[i];
  }



inline
ushort
philox_get_elem_ushort(uint* philox_state, uint* aux, const UWORD i)
  {
  return ((ushort*) philox_state)[i];
  }



inline
uint
philox_get_elem_uint(uint* philox_state, uint* aux, const UWORD i)
  {
  return philox_state[i];
  }



inline
ulong
philox_get_elem_ulong(uint* philox_state, uint* aux, const UWORD i)
  {
  if (i <= 1)
    return ((ulong*) philox_state)[i];
  else
    return ((ulong*) aux)[i - 2];
  }



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
    if (isnan(sqrt_inner) || isinf(sqrt_inner))
      sqrt_inner = (fp_eT1) 0;
    fp_eT1 trig_inner = 2 * M_PI * (COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 1) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());

    mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * cos(trig_inner)) * sd + mu);
    i += num_threads;
    if (i < n)
      mem[mem_offset + i] = (eT1) ((sqrt(sqrt_inner) * sin(trig_inner)) * sd + mu);
    i += num_threads;

    sqrt_inner = -2 * log(COOT_FN(philox_get_elem_,uint_eT1)(local_philox_state, aux_mem, 2) / (fp_eT1) COOT_FN(coot_type_max_,uint_eT1)());
    if (isnan(sqrt_inner) || isinf(sqrt_inner))
      sqrt_inner = (fp_eT1) 0;
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
