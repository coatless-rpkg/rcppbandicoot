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
COOT_FN(PREFIX,trace)(eT1* out,
                      const eT1* A,
                      const UWORD n_rows,
                      const UWORD N)
  {
  const UWORD id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id == 0)
    {
    eT1 acc = TO_ET1(0);
    // runtime unrolling is not supported by CUDA
    for(UWORD i=0; i<N; ++i)
      {
      acc += A[i + i*n_rows];
      }
    out[0] = acc;
    }
  }
