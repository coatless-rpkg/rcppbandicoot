// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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
COOT_FN(PREFIX,shuffle_large)(__global eT1* out,
                              const UWORD out_offset,
                              const UWORD out_incr, /* how many eT1s to advance to get to the start of the next element to shuffle */
                              const UWORD out_elem_stride, /* how many eT1s between each eT1 in each element */
                              __global const eT1* in,
                              const UWORD in_offset,
                              const UWORD in_incr,
                              const UWORD in_elem_stride,
                              __global const UWORD* block_offsets,
                              const UWORD n_elem,
                              const UWORD elems_per_elem,
                              const UWORD n_elem_pow2,
                              __global const UWORD* philox_key,
                              const UWORD num_bits,
                              __local volatile UWORD* aux_mem)
  {
  const UWORD tid = get_global_id(0);
  const UWORD local_tid = get_local_id(0);
  const UWORD local_size = get_local_size(0);

  // Recompute our bijective shuffle location.
  const UWORD in_loc = var_philox(tid, philox_key, num_bits);

  // We actually have to perform the up-sweep a second time, since we did not save the memory the first time.
  aux_mem[local_tid] = (in_loc < n_elem);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Now, prefix-sum the auxiliary memory for this block.
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
    aux_mem[local_size - 1] = 0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = 1; s <= local_size / 2; s *= 2)
    {
    offset >>= 1;
    if (local_tid < s)
      {
      const UWORD ai = offset * (2 * local_tid + 1) - 1;
      const UWORD bi = offset * (2 * local_tid + 2) - 1;
      UWORD tmp = aux_mem[ai];
      aux_mem[ai] = aux_mem[bi];
      aux_mem[bi] += tmp;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  // With the prefix sum complete, we shuffle our result into position aux_mem[tid], but only if we are a thread with a "valid" output.
  if (in_loc < n_elem)
    {
    const UWORD in_addr_offset = in_offset + in_loc * in_incr;
    const UWORD out_addr_offset = out_offset + (aux_mem[local_tid] + block_offsets[get_group_id(0)]) * out_incr;

    for (UWORD i = 0; i < elems_per_elem; ++i)
      {
      out[out_addr_offset + (i * out_elem_stride)] = in[in_addr_offset + (i * in_elem_stride)];
      }
    }
  }
