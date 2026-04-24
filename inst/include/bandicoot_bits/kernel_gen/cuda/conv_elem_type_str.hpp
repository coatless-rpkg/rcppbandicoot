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
// `conv_elem_type_str` specialization for CUDA
//


struct conv_elem_type_prefix
  {
  static inline constexpr size_t len() { return 8; }
  static inline constexpr char_array<9> str() { return char_array<9>{ "coot_to_" }; }
  };



template<typename eT>
struct conv_elem_type_str< eT, CUDA_BACKEND > : public concat_str
  <
  conv_elem_type_prefix,
  elem_type_str<eT, CUDA_BACKEND>
  > { };
