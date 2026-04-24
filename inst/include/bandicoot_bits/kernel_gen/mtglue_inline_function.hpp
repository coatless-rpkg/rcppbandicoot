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
// `mtglue_inline_function` is a wrapper class that defines an inline function with
// the given body as a compile-time string, used for elementwise operations that convert
// the output type.
//
// The first argument to this function, the scalar we are performing the function on,
// is always called "x".  The second argument is always called "y".
//
// Example:
//
// mtglue_inline_function<uword, float, CUDA_BACKEND, coot_lt_name, coot_lt_body>
//
// could reasonably yield the definition
//
// inline uword coot_lt(const float x, const float y) { return x < y; }
//

template
  <
  typename out_eT,                          // element type function should return
  typename eT,                              // element type function should accept
  coot_backend_t backend,                   // backend we are generating the function for
  typename func_name,                       // name of the function (e.g. "coot_lt")
  typename func_body                        // body of function (without "return")
  >
using mtglue_inline_function = concat_str
  <
  func_prefix<backend>,                // optional __device__ or similar
  eop_inline_str<backend>,             // inline
  elem_type_str<out_eT, backend>,      // <eT>
  space,                               //
  func_name,                           // func_name
  func_name_suffix<out_eT, backend>,   // optional "_uword" for some backends
  func_name_suffix<eT, backend>,       // optional "_float" for some backends
  eop_inline_function_arg,             // (const
  elem_type_str<eT, backend>,          // <eT>
  space_x,                             //  x
  eop_extra_arg_list_const,            // , const
  elem_type_str<eT, backend>,          // <eT>
  space_y,                             //  y
  eop_inline_function_body,            // ) { return
  conv_elem_type_str<out_eT, backend>, // out_eT
  open_paren,                          // (
  func_body,                           // whatever the function body is
  close_paren,                         // )
  semicolon_close                      // ; } (plus a newline)
  >;
