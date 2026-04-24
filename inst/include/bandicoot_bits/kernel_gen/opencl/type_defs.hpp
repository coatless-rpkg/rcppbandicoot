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
// definitions of source strings required for supporting individual types
// in OpenCL kernels
//

template<>
struct type_def<CL_BACKEND, u8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u8_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, s8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s8_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, u16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u16_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, s16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s16_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, u32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u32_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, s32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s32_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, u64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u64_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, s64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s64_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, fp16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/h_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, float>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/f_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, double>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/d_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, std::complex<float> >
  {
  static inline constexpr auto& src_str() { return
      #include "defs/c_defs.cl"
      ; }
  };



template<>
struct type_def<CL_BACKEND, std::complex<double> >
  {
  static inline constexpr auto& src_str() { return
      #include "defs/z_defs.cl"
      ; }
  };
