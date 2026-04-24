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
// definitions of all kernel source locations for generated kernels
//



template<>
struct kernel_src_str< kernel_id::fill, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/fill.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::copy, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/copy.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::sum_colwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_colwise.cl"
      ; }
  };

template<>
struct kernel_src_str< kernel_id::mean_colwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_colwise.cl"
      ; }
  };

template<>
struct kernel_src_str< kernel_id::max_colwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_colwise.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::min_colwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_colwise.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::sum_rowwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_rowwise.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::mean_rowwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_rowwise.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::max_rowwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_rowwise.cl"
      ; }
  };



template<>
struct kernel_src_str< kernel_id::min_rowwise, CL_BACKEND >
  {
  static inline constexpr auto& src_str() { return
      #include "kernels/reduce_rowwise.cl"
      ; }
  };
