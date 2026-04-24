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
// template metaprogramming utilities for concatenating strings in macros
//



template<coot_backend_t backend>
struct macro_defn
  {
  struct prefix
    {
    static inline constexpr size_t len() { return 3; }
    static inline constexpr char_array<4> str() { return char_array<4>{ "-D " }; }
    };

  struct suffix
    {
    static inline constexpr size_t len() { return 0; }
    static inline constexpr char_array<1> str() { return char_array<1>{ "" }; }
    };
  };



// OpenCL requires escaped macros

template<>
struct macro_defn< CL_BACKEND >
  {
  struct prefix
    {
    static inline constexpr size_t len() { return 4; }
    static inline constexpr char_array<5> str() { return char_array<5>{ "-D \"" }; }
    };

  struct suffix
    {
    static inline constexpr size_t len() { return 1; }
    static inline constexpr char_array<2> str() { return char_array<2>{ "\"" }; }
    };
  };
