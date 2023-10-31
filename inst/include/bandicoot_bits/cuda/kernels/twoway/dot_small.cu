// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

__global__
void
COOT_FN(PREFIX,dot_small)(twoway_promoted_eT* out_mem,
                          const eT1* A,
                          const eT2* B,
                          const UWORD n_elem)
  {
  twoway_promoted_eT* aux_mem = (twoway_promoted_eT*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  aux_mem[tid] = 0;

  while (i + blockDim.x < n_elem)
    {
    const twoway_promoted_eT A_i1 = (twoway_promoted_eT) A[i];
    const twoway_promoted_eT B_i1 = (twoway_promoted_eT) B[i];

    const twoway_promoted_eT A_i2 = (twoway_promoted_eT) A[i + blockDim.x];
    const twoway_promoted_eT B_i2 = (twoway_promoted_eT) B[i + blockDim.x];

    // copy to local shared memory
    aux_mem[tid] += (A_i1 * B_i1) + (A_i2 * B_i2);
    i += grid_size;
    }
  if (i < n_elem)
    {
    const twoway_promoted_eT A_i1 = (twoway_promoted_eT) A[i];
    const twoway_promoted_eT B_i1 = (twoway_promoted_eT) B[i];

    aux_mem[tid] += (A_i1 * B_i1);
    }

  for (UWORD s = blockDim.x / 2; s > 0; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
