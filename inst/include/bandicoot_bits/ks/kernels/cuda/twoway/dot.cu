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



__device__
void
COOT_FN(PREFIX,dot_subgroup_reduce)(volatile twoway_promoted_eT* data, int tid)
  {
  data[tid] += data[tid + 32];
  data[tid] += data[tid + 16];
  data[tid] += data[tid + 8];
  data[tid] += data[tid + 4];
  data[tid] += data[tid + 2];
  data[tid] += data[tid + 1];
  }



// this kernel is technically incorrect if the size is not a factor of 2!
__global__
void
COOT_FN(PREFIX,dot)(twoway_promoted_eT* out_mem,
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
    const twoway_promoted_eT A_i1 = TO_TWOWAY_PROMOTED_ET(A[i]);
    const twoway_promoted_eT B_i1 = TO_TWOWAY_PROMOTED_ET(B[i]);

    const twoway_promoted_eT A_i2 = TO_TWOWAY_PROMOTED_ET(A[i + blockDim.x]);
    const twoway_promoted_eT B_i2 = TO_TWOWAY_PROMOTED_ET(B[i + blockDim.x]);

    aux_mem[tid] += (A_i1 * B_i1) + (A_i2 * B_i2); // copy to local shared memory
    i += grid_size;
    }
  if (i < n_elem)
    {
    const twoway_promoted_eT A_i1 = TO_TWOWAY_PROMOTED_ET(A[i]);
    const twoway_promoted_eT B_i1 = TO_TWOWAY_PROMOTED_ET(B[i]);

    aux_mem[tid] += (A_i1 * B_i1);
    }
  __syncthreads();

  for (UWORD s = blockDim.x / 2; s > 32; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    __syncthreads();
    }

  if (tid < 32) // unroll last warp's worth of work
    {
    COOT_FN(PREFIX,dot_subgroup_reduce)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
