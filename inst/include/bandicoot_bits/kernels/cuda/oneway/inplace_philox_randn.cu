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

// This kernel is a placeholder---we don't use it with the CUDA backend.
// It does nothing.  We use cuRand instead.

__global__
void
COOT_FN(PREFIX,inplace_philox_randn)(eT1* mem,
                                     unsigned int* philox_state,
                                     const UWORD n_elem,
                                     const fp_eT1 mu,
                                     const fp_eT1 sd)
  {
  // Do nothing!
  }
