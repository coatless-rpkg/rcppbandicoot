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
// `eop_inline_function` is a wrapper class that defines an inline function with
// the given body as a compile-time string, used for elementwise operations.
//
// The first argument to this function, the scalar we are performing the function on,
// is always called "x".  Extra arguments with the given names will be added if
// num_extra_args > 0 and they will be given the names in `arg_names`.
//
// Example:
//
// eop_inline_function<float, CUDA_BACKEND, coot_neg_name, 0, coot_neg_args, coot_neg_body>
//
// could reasonably yield the definition
//
// inline float coot_neg(const float x) { return -x; }
//

template<coot_backend_t backend>
struct eop_inline_str { static inline constexpr auto& str() { return "inline "; } };

template<typename eT, coot_backend_t backend, size_t num_extra_args, template<size_t arg_num> class arg_names>
struct eop_extra_arg_list : public concat_str<> { };

template<typename eT, coot_backend_t backend, template<size_t arg_num> class arg_names>
struct eop_extra_arg_list<eT, backend, 1, arg_names> : public concat_str
  <
  eop_extra_arg_list_const,   // , const
  elem_type_str<eT, backend>, // <eT>
  space,                      //
  arg_names<0>                // arg_name
  > { };

template<typename eT, coot_backend_t backend, template<size_t arg_num> class arg_names>
struct eop_extra_arg_list<eT, backend, 2, arg_names> : public concat_str
  <
  eop_extra_arg_list_const,   // , const
  elem_type_str<eT, backend>, // <eT>
  space,                      //
  arg_names<0>,               // arg_name1
  eop_extra_arg_list_const,   // , const
  elem_type_str<eT, backend>, // <eT>
  space,                      //
  arg_names<1>                // arg_name2
  > { };



template
  <
  typename eT,                              // element type function should accept and return
  coot_backend_t backend,                   // backend we are generating the function for
  typename func_name,                       // name of the function (e.g. "coot_neg")
  size_t num_extra_args,                    // number of extra arguments after the scalar "x"
  template<size_t arg_num> class arg_names, // names to use for each extra argument
  typename func_body                        // body of function (without "return")
  >
using eop_inline_function = concat_str
  <
  func_prefix<backend>,          // optional __device__ or similar
  eop_inline_str<backend>,       // inline (omitted for Vulkan/GLSL, which is a reserverd construct)
  elem_type_str<eT, backend>,    // <eT>
  space,                         //
  func_name,                     // func_name
  func_name_suffix<eT, backend>, // optional "_float" for some backends
  eop_inline_function_arg,       // (const
  elem_type_str<eT, backend>,    // <eT>
  space_x,                       // x
  eop_extra_arg_list<eT, backend, num_extra_args, arg_names>, // optional , const <eT> extra_arg1, const <eT> extra_arg2
  eop_inline_function_body,      // ) { return
  func_body,                     // whatever the function body is
  semicolon_close                // ; } (and newline)
  >;
