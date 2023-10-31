// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
COOT_FN(PREFIX,trace)(__global eT1* out,
                      __global const eT1* A,
                      const UWORD n_rows,
                      const UWORD N)
  {
  const UWORD id = get_global_id(0);
  if(id == 0)
    {
    eT1 acc = (eT1)(0);
    #pragma unroll
    for(UWORD i=0; i<N; ++i)
      {
      acc += A[i + i*n_rows];
      }
    out[0] = acc;
    }
  }
