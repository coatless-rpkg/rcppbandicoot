// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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



__kernel
void
COOT_FN(PREFIX,extract_sve1)(__global eT2* out_mem,
                             const UWORD out_mem_offset,
                             __global const eT1* in_mem,
                             const UWORD in_mem_offset,
                             __global const UWORD* in_locs,
                             const UWORD in_locs_offset,
                             const UWORD n_elem)
  {
  const UWORD i = get_global_id(0);

  if (i < n_elem)
    {
    out_mem[i + out_mem_offset] = (eT2) in_mem[in_locs[i + in_locs_offset] + in_mem_offset];
    }
  }
