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
struct elem_type_str< u8, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 7; }
  static inline constexpr char_array<8> str() { return char_array<8>{ "uint8_t" }; }
  };



template<>
struct elem_type_str< s8, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 6; }
  static inline constexpr char_array<7> str() { return char_array<7>{ "int8_t" }; }
  };



template<>
struct elem_type_str< u16, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 8; }
  static inline constexpr char_array<9> str() { return char_array<9>{ "uint16_t" }; }
  };



template<>
struct elem_type_str< s16, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 7; }
  static inline constexpr char_array<8> str() { return char_array<8>{ "int16_t" }; }
  };



template<>
struct elem_type_str< u64, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 8; }
  static inline constexpr char_array<9> str() { return char_array<9>{ "uint64_t" }; }
  };



template<>
struct elem_type_str< s64, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 7; }
  static inline constexpr char_array<8> str() { return char_array<8>{ "int64_t" }; }
  };



template<>
struct elem_type_str< fp16, VULKAN_BACKEND >
  {
  static inline constexpr size_t len() { return 9; }
  static inline constexpr char_array<10> str() { return char_array<10>{ "float16_t" }; }
  };
