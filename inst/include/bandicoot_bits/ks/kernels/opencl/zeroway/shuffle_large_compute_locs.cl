// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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



// This performs the first part of the shuffle_vec kernel: it computes random
// locations for the output using the variable philox bijective shuffle,
// and then does the first step of the output compression (the upsweep of the
// shifted prefix sum).
__kernel
void
shuffle_large_compute_locs(__global UWORD* out_block_mem,
                           const UWORD n_elem,
                           const UWORD n_elem_pow2,
                           __global const UWORD* philox_key,
                           const UWORD num_bits,
                           __local volatile UWORD* aux_mem)
  {
  const UWORD tid = get_global_id(0);
  const UWORD local_tid = get_local_id(0);
  const UWORD local_size = get_local_size(0);

  // Get our bijective shuffle location.
  const UWORD in_loc = var_philox(tid, philox_key, num_bits);

  // Fill aux_mem with the indicator of whether we are out of bounds.
  // Then, we'll prefix-sum it.  This will tell us where to put our result.
  aux_mem[local_tid] = (in_loc < n_elem);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Now, prefix-sum the auxiliary memory.
  // This allows us to do the shuffle-compaction step.
  UWORD offset = 1;
  for (UWORD s = local_size / 2; s > 0; s >>= 1)
    {
    if (local_tid < s)
      {
      const UWORD ai = offset * (2 * local_tid + 1) - 1;
      const UWORD bi = offset * (2 * local_tid + 2) - 1;
      aux_mem[bi] += aux_mem[ai];
      }
    offset *= 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (local_tid == 0)
    {
    out_block_mem[get_group_id(0)] = aux_mem[local_size - 1];
    }
  }
