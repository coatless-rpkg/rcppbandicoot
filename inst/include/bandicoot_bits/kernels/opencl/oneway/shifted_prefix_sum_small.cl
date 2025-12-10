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



// This kernel performs shifted prefix-sum on `mem` assuming that (2 * local group size) <= n_elem.
// It's okay if n_elem is not a power of 2.
__kernel
void
COOT_FN(PREFIX,shifted_prefix_sum_small)(__global eT1* mem,
                                         const UWORD global_mem_offset,
                                         const UWORD n_elem,
                                         __local volatile eT1* aux_mem)
  {
  const UWORD local_tid = get_local_id(0);
  const UWORD local_size = get_local_size(0); // will be the same across all workgroups (by calling convention), and must be a power of 2
  const UWORD group_id = get_group_id(0);

  // Copy relevant memory to auxiliary memory.
  // This workgroup is responsible for mem[group_id * (2 * local_size)] to mem[(group_id + 1) * (2 * local_size) - 1].
  const UWORD group_offset = group_id * (2 * local_size);
  const UWORD mem_offset   = group_offset + 2 * local_tid;

  aux_mem[mem_offset    ] = (mem_offset     < n_elem) ? mem[global_mem_offset + mem_offset    ] : (eT1) 0;
  aux_mem[mem_offset + 1] = (mem_offset + 1 < n_elem) ? mem[global_mem_offset + mem_offset + 1] : (eT1) 0;

  UWORD offset = 1;
  for (UWORD s = local_size; s > 0; s >>= 1)
    {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid < s)
      {
      const UWORD ai = group_offset + offset * (2 * local_tid + 1) - 1;
      const UWORD bi = group_offset + offset * (2 * local_tid + 2) - 1;
      aux_mem[bi] += aux_mem[ai];
      }
    offset *= 2;
    }

  // Prepare for down-sweep by setting the last element to 0.
  if (local_tid == 0)
    {
    aux_mem[2 * local_size - 1] = 0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = 1; s <= local_size; s *= 2)
    {
    offset >>= 1;
    if (local_tid < s)
      {
      const UWORD ai = group_offset + offset * (2 * local_tid + 1) - 1;
      const UWORD bi = group_offset + offset * (2 * local_tid + 2) - 1;
      eT1 tmp = aux_mem[ai];
      aux_mem[ai] = aux_mem[bi];
      aux_mem[bi] += tmp;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  // Copy results back to memory.
  if (mem_offset + 1 < n_elem)
    {
    mem[global_mem_offset + mem_offset    ] = aux_mem[mem_offset    ];
    mem[global_mem_offset + mem_offset + 1] = aux_mem[mem_offset + 1];
    }
  else if (mem_offset < n_elem)
    {
    mem[global_mem_offset + mem_offset    ] = aux_mem[mem_offset    ];
    }
  }
