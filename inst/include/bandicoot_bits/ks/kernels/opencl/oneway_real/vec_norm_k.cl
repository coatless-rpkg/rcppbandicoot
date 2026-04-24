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
COOT_FN(PREFIX,vec_norm_k)(__global const eT1* in_mem,
                           const UWORD in_mem_offset,
                           const UWORD n_elem,
                           __global eT1* out_mem,
                           const UWORD out_mem_offset,
                           __local volatile eT1* aux_mem,
                           const UWORD k)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 0;

  while (i + get_local_size(0) < n_elem)
    {
    const eT1 v1 = pow(in_mem[in_mem_offset + i],                     (eT1) k);
    const eT1 v2 = pow(in_mem[in_mem_offset + i + get_local_size(0)], (eT1) k);
    aux_mem[tid] += v1 + v2;
    i += grid_size;
    }
  if (i < n_elem)
    {
    const eT1 v1 = pow(in_mem[in_mem_offset + i], (eT1) k);
    aux_mem[tid] += v1;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > SUBGROUP_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < SUBGROUP_SIZE)
    {
    COOT_FN_3(PREFIX,accu_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[out_mem_offset + get_group_id(0)] = aux_mem[0];
    }
  }
