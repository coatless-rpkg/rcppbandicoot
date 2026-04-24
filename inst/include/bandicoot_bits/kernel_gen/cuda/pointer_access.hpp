// Copyright 2026 Ryan Curtin (http://www.ratml.org)
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
// coot_pointer_access defines the prefix and suffix for accessing a pointer argument.
//

template<typename arg_name_prefix>
struct coot_pointer_access<CUDA_BACKEND, arg_name_prefix>
  {
  struct internal_prefix
    {
    static inline constexpr size_t len() { return 17; }
    static inline constexpr char_array<18> str() { return char_array<18>{ "COOT_CONCAT(name," }; }
    };

  struct internal_midfix
    {
    static inline constexpr size_t len() { return 6; }
    static inline constexpr char_array<7> str() { return char_array<7>{ "_ptr)[" }; }
    };



  using prefix = concat_str<internal_prefix, arg_name_prefix, internal_midfix>;

  struct suffix
    {
    static inline constexpr size_t len() { return 1; }
    static inline constexpr char_array<2> str() { return char_array<2>{ "]" }; }
    };
  };
