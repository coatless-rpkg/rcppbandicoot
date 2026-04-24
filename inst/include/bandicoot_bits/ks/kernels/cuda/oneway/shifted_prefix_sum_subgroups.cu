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



// This kernel performs the shifted prefix-sum on each individual block.
// This is the same as just running a regular prefix-sum kernel, except that
// `out_mem[i]` will store the total sum of elements in block `i`.
// After running this, to finish prefix-sum on the entire memory, offsets for
// each workgroup need to be added.
__global__
void
COOT_FN(PREFIX,shifted_prefix_sum_subgroups)(eT1* mem,
                                             eT1* out_mem,
                                             const UWORD n_elem)
  {
  eT1* aux_mem = (eT1*) aux_shared_mem;

  const UWORD local_tid = threadIdx.x;
  const UWORD local_size = blockDim.x; // will be the same across all workgroups (by calling convention), and must be a power of 2
  const UWORD group_id = blockIdx.x;

  // Copy relevant memory to auxiliary memory.
  // This workgroup is responsible for mem[group_id * (2 * local_size)] to mem[(group_id + 1) * (2 * local_size) - 1].
  const UWORD group_offset = group_id * (2 * local_size);
  const UWORD local_offset = 2 * local_tid;
  const UWORD mem_offset   = group_offset + local_offset;

  aux_mem[local_offset    ] = (mem_offset     < n_elem) ? mem[mem_offset    ] : TO_ET1(0);
  aux_mem[local_offset + 1] = (mem_offset + 1 < n_elem) ? mem[mem_offset + 1] : TO_ET1(0);

  UWORD offset = 1;
  for (UWORD s = local_size; s > 0; s >>= 1)
    {
    if (local_tid < s)
      {
      const UWORD ai = offset * (local_offset + 1) - 1;
      const UWORD bi = offset * (local_offset + 2) - 1;
      aux_mem[bi] += aux_mem[ai];
      }
    offset *= 2;
    __syncthreads();
    }

  if (mem_offset + 1 < n_elem)
    {
    mem[mem_offset    ] = aux_mem[local_offset    ];
    mem[mem_offset + 1] = aux_mem[local_offset + 1];
    }
  else if (mem_offset < n_elem)
    {
    mem[mem_offset    ] = aux_mem[local_offset    ];
    }

  if (local_tid == 0)
    {
    // Write the sum of the subarray to the output memory.
    out_mem[group_id] = aux_mem[2 * local_size - 1];
    // Prepare for the downsweep.
    aux_mem[2 * local_size - 1] = 0;
    }
  __syncthreads();

  offset = local_size;
  for (UWORD s = 1; s <= local_size; s *= 2)
    {
    if (local_tid < s)
      {
      const UWORD ai = offset * (local_offset + 1) - 1;
      const UWORD bi = offset * (local_offset + 2) - 1;
      eT1 tmp = aux_mem[ai];
      aux_mem[ai] = aux_mem[bi];
      aux_mem[bi] += tmp;
      }
    offset >>= 1;
    __syncthreads();
    }

  // Copy results back to memory.
  // The results here are the prefix-summed results for each individual
  // workgroup.
  if (mem_offset + 1 < n_elem)
    {
    mem[mem_offset    ] = aux_mem[local_offset    ];
    mem[mem_offset + 1] = aux_mem[local_offset + 1];
    }
  else if (mem_offset < n_elem)
    {
    mem[mem_offset    ] = aux_mem[local_offset    ];
    }
  }
