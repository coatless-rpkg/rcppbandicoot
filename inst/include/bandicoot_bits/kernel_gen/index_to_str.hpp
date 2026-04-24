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
// template metaprogramming utilities to convert a size_t into a string at compile time
//



// fail for any non-explicit specialization
template<size_t i>
struct index_to_str { };



// perhaps a little brutalistic and inelegant, but we don't have kernels with all that many arguments...

template<>
struct index_to_str<0>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "0" }; }
  };



template<>
struct index_to_str<1>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "1" }; }
  };



template<>
struct index_to_str<2>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "2" }; }
  };



template<>
struct index_to_str<3>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "3" }; }
  };



template<>
struct index_to_str<4>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "4" }; }
  };



template<>
struct index_to_str<5>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "5" }; }
  };



template<>
struct index_to_str<6>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "6" }; }
  };



template<>
struct index_to_str<7>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "7" }; }
  };



template<>
struct index_to_str<8>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "8" }; }
  };



template<>
struct index_to_str<9>
  {
  static inline constexpr size_t len() { return 1; }
  static inline constexpr char_array<2> str() { return char_array<2>{ "9" }; }
  };
