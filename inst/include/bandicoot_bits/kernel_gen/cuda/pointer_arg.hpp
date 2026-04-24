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
// coot_pointer_arg defines a pointer argument for a given backend.
// For the OpenCL backend, this actually defines two arguments since the offset
// must be a separate argument.
//

struct coot_cuda_pointer_arg_macro
  {
  static inline constexpr size_t len() { return 19; }
  static inline constexpr char_array<20> str() { return char_array<20>{ "* COOT_CONCAT(name," }; }
  };

struct coot_cuda_pointer_ptr_name
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<5> str() { return char_array<5>{ "_ptr" }; }
  };

template<typename eT, typename arg_name_prefix>
struct coot_pointer_arg<eT, CUDA_BACKEND, arg_name_prefix> : public concat_str
  <
  elem_type_str<eT, CUDA_BACKEND>,
  coot_cuda_pointer_arg_macro,
  arg_name_prefix,
  coot_cuda_pointer_ptr_name,
  close_paren
  > { };
