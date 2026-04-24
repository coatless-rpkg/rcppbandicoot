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
// `elem_type_str` defines the string identifier of a given Bandicoot type on
// the device.  For most types, the identifier is the same across backends and
// is defined here.  For backend-specific types, this struct is specialized for
// each backend in
// cuda/elem_type_str.hpp and opencl/elem_type_str.hpp.
//

template<typename eT, coot_backend_t backend>
struct elem_type_str { };



template<coot_backend_t backend>
struct elem_type_str< u8, backend >
  {
  static inline constexpr size_t len() { return 5; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "uchar" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< s8, backend >
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "char" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< u16, backend >
  {
  static inline constexpr size_t len() { return 6; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "ushort" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< s16, backend >
  {
  static inline constexpr size_t len() { return 5; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "short" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< u32, backend >
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "uint" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< s32, backend >
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "int" }; }
  };



// u64 is device-specific!



template<coot_backend_t backend>
struct elem_type_str< s64, backend >
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "long" }; }
  };



// fp16 is device-specific!


template<coot_backend_t backend>
struct elem_type_str< float, backend >
  {
  static inline constexpr size_t len() { return 5; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "float" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< double, backend >
  {
  static inline constexpr size_t len() { return 6; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "double" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< std::complex<float>, backend >
  {
  static inline constexpr size_t len() { return 8; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "cx_float" }; }
  };



template<coot_backend_t backend>
struct elem_type_str< std::complex<double>, backend >
  {
  static inline constexpr size_t len() { return 9; }
  static inline constexpr char_array<len() + 1> str() { return char_array<len() + 1>{ "cx_double" }; }
  };
