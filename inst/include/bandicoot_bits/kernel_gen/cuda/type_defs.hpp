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
// in CUDA kernels
//

template<>
struct type_def<CUDA_BACKEND, u8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u8_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, s8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s8_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, u16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u16_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, s16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s16_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, u32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u32_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, s32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s32_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, u64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u64_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, s64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s64_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, fp16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/h_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, float>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/f_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, double>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/d_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, std::complex<float> >
  {
  // also include float definitions
  static inline constexpr auto& src_str() { return
      #include "defs/c_defs.cu"
      ; }
  };



template<>
struct type_def<CUDA_BACKEND, std::complex<double> >
  {
  // also include double definitions
  static inline constexpr auto& src_str() { return
      #include "defs/z_defs.cu"
      ; }
  };
