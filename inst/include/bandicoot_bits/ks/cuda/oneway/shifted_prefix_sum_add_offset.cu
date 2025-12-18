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



// This kernel adds block-specific offsets to blocks of local memory.
// Specifically, block i, which has t threads, adds offsets[i] to the range
// mem[i * (2 * t)] to mem[(i + 1) * (2 * t) - 1] (inclusive).
__global__
void
COOT_FN(PREFIX,shifted_prefix_sum_add_offset)(eT1* mem,
                                              const eT1* offsets,
                                              const UWORD n_elem)
  {
  const UWORD local_tid = threadIdx.x;
  const UWORD local_size = blockDim.x; // will be the same across all workgroups (by calling convention), and must be a power of 2
  const UWORD group_id = blockIdx.x;

  // This workgroup is responsible for mem[group_id * (2 * local_size)] to mem[(group_id + 1) * (2 * local_size) - 1].
  const UWORD group_offset = group_id * (2 * local_size);
  const UWORD local_offset = 2 * local_tid;
  const UWORD mem_offset   = group_offset + local_offset;

  const eT1 offset = offsets[group_id];

  const eT1 in_val1 = (mem_offset     < n_elem) ? mem[mem_offset    ] : TO_ET1(0);
  const eT1 in_val2 = (mem_offset + 1 < n_elem) ? mem[mem_offset + 1] : TO_ET1(0);

  const eT1 out_val1 = in_val1 + offset;
  const eT1 out_val2 = in_val2 + offset;

  // Copy results back to memory.
  if (mem_offset + 1 < n_elem)
    {
    mem[mem_offset    ] = out_val1;
    mem[mem_offset + 1] = out_val2;
    }
  else if (mem_offset < n_elem)
    {
    mem[mem_offset    ] = out_val1;
    }
  }
