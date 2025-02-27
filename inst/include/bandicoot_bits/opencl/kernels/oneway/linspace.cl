// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,linspace)(__global eT1* out_mem,
                         const UWORD out_mem_offset,
                         const UWORD mem_incr,
                         const eT1 start,
                         const eT1 end,
                         const eT1 step,
                         const UWORD num)
  {
  const UWORD idx = get_global_id(0);
  if (idx < num)
    {
    out_mem[out_mem_offset + idx * mem_incr] = (eT1) (start + step * idx);
    }
  }
