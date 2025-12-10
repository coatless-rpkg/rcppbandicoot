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
COOT_FN(PREFIX,rel_any_nonfinite)(__global const eT1* X,
                                  const UWORD X_offset,
                                  const UWORD n_elem,
                                  __global uint* out,
                                  const UWORD out_offset,
                                  __local volatile uint* aux_mem,
                                  const eT1 val /* ignored */)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 0;

  while (i + get_local_size(0) < n_elem)
    {
    const eT1 val1 = X[X_offset + i];
    const eT1 val2 = X[X_offset + i + get_local_size(0)];

    aux_mem[tid] |= isnan(val1) | isinf(val1);
    aux_mem[tid] |= isnan(val2) | isinf(val2);
    if (aux_mem[tid] == 1)
      break;
    i += grid_size;
    }

  if (i < n_elem && aux_mem[tid] == 0)
    {
    const eT1 val1 = X[X_offset + i];
    aux_mem[tid] |= isnan(val1) | isinf(val1);
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
    COOT_FN(COOT_FN(or_subgroup_reduce_,SUBGROUP_SIZE_NAME),_u32)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out[out_offset + get_group_id(0)] = aux_mem[0];
    }
  }
