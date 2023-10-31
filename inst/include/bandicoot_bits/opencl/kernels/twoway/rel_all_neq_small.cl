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
COOT_FN(PREFIX,rel_all_neq_small)(__global const eT1* X,
                                  const UWORD n_elem,
                                  __global uint* out,
                                  __local volatile uint* aux_mem,
                                  const eT2 val)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 1;

  while (i + get_local_size(0) < n_elem)
    {
    const eT2 val1 = (eT2) X[i];
    const eT2 val2 = (eT2) X[i + get_local_size(0)];

    aux_mem[tid] &= (val1 != val);
    aux_mem[tid] &= (val2 != val);
    i += grid_size;
    }
  if (i < n_elem)
    {
    const eT2 val1 = (eT2) X[i];
    aux_mem[tid] &= (val1 != val);
    }

  for (UWORD s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);

    if (tid < s)
      {
      aux_mem[tid] &= aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out[get_group_id(0)] = aux_mem[0];
    }
  }
