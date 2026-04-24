// Copyright 2026 Marcus Edel (http://www.kurg.org)
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



template<>
struct type_def<VULKAN_BACKEND, u8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u8_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, s8>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s8_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, u16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u16_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, s16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s16_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, fp16>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/fp16_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, float>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/f_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, double>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/d_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, std::complex<float> >
  {
  static inline constexpr auto& src_str() { return
      #include "defs/cx_f_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, std::complex<double> >
  {
  static inline constexpr auto& src_str() { return
      #include "defs/cx_d_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, u32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u32_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, s32>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s32_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, u64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/u64_defs.glsl"
      ; }
  };



template<>
struct type_def<VULKAN_BACKEND, s64>
  {
  static inline constexpr auto& src_str() { return
      #include "defs/s64_defs.glsl"
      ; }
  };
