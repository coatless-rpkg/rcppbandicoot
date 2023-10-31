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

// utility structure to hold kernels (or hold structures that hold other kernels)

template<typename HeldType>
struct kernels_t
  {
  HeldType   u32_kernels;
  HeldType   s32_kernels;
  HeldType   u64_kernels;
  HeldType   s64_kernels;
  HeldType     f_kernels;
  HeldType     d_kernels;
  };
