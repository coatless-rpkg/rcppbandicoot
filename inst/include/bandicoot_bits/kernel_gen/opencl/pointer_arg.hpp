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

struct coot_cl_pointer_arg_prefix          { static inline constexpr auto& str() { return "__global ";                  } };
struct coot_cl_pointer_arg_macro1          { static inline constexpr auto& str() { return "* COOT_CONCAT(name,";        } };
struct coot_cl_pointer_arg_ptr_name        { static inline constexpr auto& str() { return "_ptr";                       } };
struct coot_cl_pointer_arg_macro2          { static inline constexpr auto& str() { return "), UWORD COOT_CONCAT(name,"; } };
struct coot_cl_pointer_arg_ptr_offset_name { static inline constexpr auto& str() { return "_ptr_offset";                } };

template<typename eT, typename arg_name_prefix>
struct coot_pointer_arg<eT, CL_BACKEND, arg_name_prefix> : public concat_str
  <
  coot_cl_pointer_arg_prefix,
  elem_type_str<eT, CL_BACKEND>,
  coot_cl_pointer_arg_macro1,
  arg_name_prefix,
  coot_cl_pointer_arg_ptr_name,
  coot_cl_pointer_arg_macro2,
  arg_name_prefix,
  coot_cl_pointer_arg_ptr_offset_name,
  close_paren
  > { };
