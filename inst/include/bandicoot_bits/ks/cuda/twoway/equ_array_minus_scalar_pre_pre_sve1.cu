// Copyright 2019-2025 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,equ_array_minus_scalar_pre_pre_sve1)(eT2* dest,
                                                    const UWORD* dest_locs,
                                                    const eT1* src,
                                                    const UWORD* src_locs,
                                                    const eT1 val_pre,
                                                    const eT2 val_post,
                                                    const UWORD n_elem)
  {
  (void) val_pre;

  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_elem)
    {
    dest[dest_locs[i]] = val_post - (TO_ET2(src[src_locs[i]]));
    }
  }
