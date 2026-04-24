// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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
// template metaprogramming utilities that supply the source for a given kernel at compile time
//



//
// For each backend and kernel, this must be defined with the function
//
// static inline constexpr auto& src_str() { return
//    #include "kernels/<name>.<ext>"
//    ; }
//
// (Note this particular spacing is needed because the #include must be on its own line.)
//
// This struct is specialized accordingly in cuda/kernels.hpp and opencl/kernels.hpp.
//

template<kernel_id::enum_id num, coot_backend_t backend>
struct kernel_src_str { };



//
// Utility struct to wrap the compile-time source string for `kernel_gen`.
//

template<kernel_id::enum_id num, coot_backend_t backend>
struct kernel_src : public kernel_src_str<num, backend>
  {

  static inline constexpr size_t len()
    {
    return sizeof(kernel_src_str<num, backend>::src_str()) - 1;
    }

  static inline constexpr char_array<len() + 1> str()
    {
    return to_array::array(kernel_src_str<num, backend>::src_str());
    }

  };
