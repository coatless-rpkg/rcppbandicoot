// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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
COOT_FN(PREFIX,ipiv_det)(__global const eT1* in_mem,
                         const UWORD in_mem_offset,
                         const UWORD n_elem,
                         __global eT1* out_mem,
                         const UWORD out_mem_offset,
                         __local volatile eT1* aux_mem)
  {
  // This kernel is not used by the OpenCL backend, so we leave it empty!
  }
