// Copyright 2019-2026 Ryan Curtin (http://www.ratml.org/)
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
R"(

__global__
void
(COOT_KERNEL_FUNC)(COOT_OBJECT_0(out),
                   COOT_OBJECT_1(in))
  {
  // Some of these may be unused depending on the type of COOT_OBJECT_0().
  const UWORD row   = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col   = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD slice = blockIdx.z * blockDim.z + threadIdx.z;

  // We expect that if it's in bounds for object 0, that it will be in bounds for object 1.
  if (COOT_OBJECT_0_BOUNDS_CHECK(out, row, col, slice))
    {
    COOT_OBJECT_0_AT(out, row, col, slice) = COOT_TO_ET0(COOT_OBJECT_1_AT(in, row, col, slice));
    }
  }

)"
