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
// macro_conv_elem_type provides a compile-time string definition of a macro defining
// the code necessary to convert to a given element type, of the form:
//
// -DCOOT_TO_ET<i>=<type>
//
// The conv_elem_type_str<eT, backend> macro must be specialized for each backend.
//

struct macro_et_conv_def  { static inline constexpr auto& str() { return "COOT_TO_ET"; } };
struct macro_from_et_str  { static inline constexpr auto& str() { return "_FROM_ET";   } };

template<typename out_eT, size_t out_i, typename in_eT, size_t in_i, coot_backend_t backend >
using macro_conv_elem_type = concat_str
  <
  typename macro_defn<backend>::prefix,
  macro_et_conv_def,
  index_to_str<out_i>,
  macro_from_et_str,
  index_to_str<in_i>,
  equals,
  conv_elem_type_str<out_eT, in_eT, backend>,
  typename macro_defn<backend>::suffix
  >;
