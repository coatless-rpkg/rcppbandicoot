// Copyright 2026 Ryan Curtin (http://www.ratml.org)
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

//
// template metaprogramming utilities that supply the source post_kernel_src for a
// given backend at compile time
//


//
// For each backend, this must be defined with the function
//
// static inline constexpr auto& src_str()
//
// This struct is specialized accordingly in cuda/post_kernel_src.hpp and opencl/post_kernel_src.hpp.
//

template<coot_backend_t backend>
struct post_kernel_src_str { };



//
// Utility struct to wrap the compile-time preamble string for `kernel_gen`.
//

template<coot_backend_t backend>
struct post_kernel_src : public concat_str< post_kernel_src_str<backend> > { };
