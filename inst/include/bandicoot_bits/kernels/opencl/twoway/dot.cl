// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
COOT_FN(PREFIX,dot_subgroup_reduce_other)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  for(UWORD i = SUBGROUP_SIZE; i > 0; i >>= 1)
    {
    if (tid < i)
      data[tid] += data[tid + i];
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
  }



void
COOT_FN(PREFIX,dot_subgroup_reduce_8)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  data[tid] += data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 1];
  }



void
COOT_FN(PREFIX,dot_subgroup_reduce_16)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  data[tid] += data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 1];
  }



void
COOT_FN(PREFIX,dot_subgroup_reduce_32)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  data[tid] += data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 1];
  }



void
COOT_FN(PREFIX,dot_subgroup_reduce_64)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  data[tid] += data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 1];
  }



void
COOT_FN(PREFIX,dot_subgroup_reduce_128)(__local volatile twoway_promoted_eT* data, UWORD tid)
  {
  data[tid] += data[tid + 128];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] += data[tid + 1];
  }



__kernel
void
COOT_FN(PREFIX,dot)(__global twoway_promoted_eT* out_mem,
                    __global const eT1* A,
                    const UWORD A_offset,
                    __global const eT2* B,
                    const UWORD B_offset,
                    const UWORD n_elem,
                    __local volatile twoway_promoted_eT* aux_mem)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 0;

  while (i + get_local_size(0) < n_elem)
    {
    aux_mem[tid] += (((twoway_promoted_eT) A[A_offset + i]) * ((twoway_promoted_eT) B[B_offset + i])) +
        (((twoway_promoted_eT) A[A_offset + i + get_local_size(0)]) * ((twoway_promoted_eT) B[B_offset + i + get_local_size(0)]));
    i += grid_size;
    }
  if (i < n_elem)
    {
    aux_mem[tid] += (((twoway_promoted_eT) A[A_offset + i]) * ((twoway_promoted_eT) B[B_offset + i]));
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
    COOT_FN_3(PREFIX,dot_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[get_group_id(0)] = aux_mem[0];
    }
  }
