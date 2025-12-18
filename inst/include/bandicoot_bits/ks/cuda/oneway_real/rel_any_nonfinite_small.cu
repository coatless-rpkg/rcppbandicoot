// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,rel_any_nonfinite_small)(const eT1* X,
                                        const UWORD n_elem,
                                        uint* out,
                                        const eT1 val /* ignored */)
  {
  uint* aux_mem = (uint*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  aux_mem[tid] = 0;

  while (i + blockDim.x < n_elem)
    {
    // copy to local shared memory
    const eT1 val1 = X[i];
    const eT1 val2 = X[i + blockDim.x];

    aux_mem[tid] |= !coot_isfinite(val1);
    aux_mem[tid] |= !coot_isfinite(val2);
    if (aux_mem[tid] == 1)
      break;

    i += grid_size;
    }

  if (i < n_elem && aux_mem[tid] == 0)
    {
    const eT1 val1 = X[i];

    aux_mem[tid] |= !coot_isfinite(val1);
    }

  for (UWORD s = blockDim.x / 2; s > 0; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] |= aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out[blockIdx.x] = aux_mem[0];
    }
  }
