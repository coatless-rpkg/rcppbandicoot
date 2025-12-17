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
COOT_FN(PREFIX,radix_sort_multi_wg_shuffle)(eT1* A,
                                            eT1* out,
                                            UWORD* counts,
                                            const UWORD n_elem,
                                            const UWORD sort_type,
                                            const UWORD start_bit)
  {
  // This kernel is a placeholder and is not used by the CUDA backend.
  }
