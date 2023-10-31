// Copyright 2021 Ryan Curtin (https://www.ratml.org/)
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
COOT_FN(PREFIX,max_abs)(__global const eT1* in_mem,
                        const UWORD n_elem,
                        __global eT1* out_mem,
                        __local volatile eT1* aux_mem)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  // Make sure all auxiliary memory is initialized to something that won't
  // screw up the final reduce.
  aux_mem[tid] = COOT_FN(coot_type_max_,eT1)();

  if (i < n_elem)
    {
    aux_mem[tid] = ET1_ABS(in_mem[i]);
    }
  if (i + get_local_size(0) < n_elem)
    {
    aux_mem[tid] = max(aux_mem[tid], (eT1) ET1_ABS(in_mem[i + get_local_size(0)]));
    }
  i += grid_size;

  while (i + get_local_size(0) < n_elem)
    {
    aux_mem[tid] = max(aux_mem[tid], (eT1) ET1_ABS(in_mem[i]));
    aux_mem[tid] = max(aux_mem[tid], (eT1) ET1_ABS(in_mem[i + get_local_size(0)]));
    i += grid_size;
    }
  if (i < n_elem)
    {
    aux_mem[tid] = max(aux_mem[tid], (eT1) ET1_ABS(in_mem[i]));
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > SUBGROUP_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] = max(aux_mem[tid], aux_mem[tid + s]);
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < SUBGROUP_SIZE)
    {
    COOT_FN_3(PREFIX,max_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[get_group_id(0)] = aux_mem[0];
    }
  }
