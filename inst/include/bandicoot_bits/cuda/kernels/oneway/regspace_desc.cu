// Copyright 2025 Ryan Curtin (http://ratml.org)
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
COOT_FN(PREFIX,regspace_desc)(eT1* out_mem,
                              const UWORD mem_incr,
                              const eT1 start,
                              const eT1 end,
                              const eT1 delta,
                              const UWORD num)
  {
  UWORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num)
    {
    out_mem[idx * mem_incr] = (eT1) start - delta * idx;
    }
  }
