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
__global__
void
COOT_FN(PREFIX,logspace)(eT1* out_mem,
                         const UWORD mem_incr,
                         const eT1 start,
                         const eT1 end,
                         const eT1 step,
                         const UWORD num)
  {
  UWORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num)
    {
    out_mem[idx * mem_incr] = TO_ET1(pow(TO_FP_ET1(10), TO_FP_ET1(start + step * (TO_ET1(idx)))));
    }
  }
