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
// macro_kernel_name<num, T> provides a compile-time string that defines a macro
// "-DCOOT_KERNEL_FUNC=<full_name>" where <full_name> is the full prefixed name
// of the kernel (see `full_name`).
//

struct macro_kernel_func_def
  {
  static inline constexpr size_t len() { return 20; }
  static inline constexpr char_array<21> str()
    {
    return char_array<21>{ "-D COOT_KERNEL_FUNC=" };
    }
  };



template<kernel_id::enum_id num, typename... Ts>
struct macro_kernel_name : public concat_str
  <
  macro_kernel_func_def,
  full_name<num, Ts...>
  > { };
