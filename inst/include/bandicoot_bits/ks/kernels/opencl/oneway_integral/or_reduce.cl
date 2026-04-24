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



void
COOT_FN(PREFIX,or_subgroup_reduce_other)(__local volatile eT1* data, UWORD tid)
  {
  for(UWORD i = SUBGROUP_SIZE; i > 0; i >>= 1)
    {
    if (tid < i)
      data[tid] |= data[tid + i];
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
  }



void
COOT_FN(PREFIX,or_subgroup_reduce_8)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
COOT_FN(PREFIX,or_subgroup_reduce_16)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
COOT_FN(PREFIX,or_subgroup_reduce_32)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
COOT_FN(PREFIX,or_subgroup_reduce_64)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] |= data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void COOT_FN(PREFIX,or_subgroup_reduce_128)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] |= data[tid + 128];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



__kernel
void
COOT_FN(PREFIX,or_reduce)(__global const eT1* in_mem,
                          const UWORD in_mem_offset,
                          const UWORD n_elem,
                          __global eT1* out_mem,
                          const UWORD out_mem_offset,
                          __local volatile eT1* aux_mem)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  // Make sure all auxiliary memory is initialized to something that won't
  // screw up the final reduce.
  aux_mem[tid] = (eT1) 0;

  while (i + get_local_size(0) < n_elem)
    {
    aux_mem[tid] |= in_mem[in_mem_offset + i];
    aux_mem[tid] |= in_mem[in_mem_offset + i + get_local_size(0)];
    i += grid_size;
    }
  if (i < n_elem)
    {
    aux_mem[tid] |= in_mem[in_mem_offset + i];
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > SUBGROUP_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] |= aux_mem[tid + s];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < SUBGROUP_SIZE)
    {
    COOT_FN_3(PREFIX,or_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[out_mem_offset + get_group_id(0)] = aux_mem[0];
    }
  }
