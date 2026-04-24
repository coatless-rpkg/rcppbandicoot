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
// `elem_type_str` specializations for CUDA-specific types
//


template<>
struct elem_type_str< u64, CUDA_BACKEND >
  {
  static inline constexpr size_t len() { return 6; }
  static inline constexpr char_array<7> str() { return char_array<7>{ "size_t" }; }
  };



template<>
struct elem_type_str< fp16, CUDA_BACKEND >
  {
  static inline constexpr size_t len() { return 6; }
  static inline constexpr char_array<7> str() { return char_array<7>{ "__half" }; }
  };
