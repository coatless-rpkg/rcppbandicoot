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
// macro_elem_access provides a compile-time string definition of a macro defining
// the given element type, of the form:
//
// -DET<i>=<type>
//

struct macro_et_def { static inline constexpr auto& str() { return "-D ET"; } };

template<typename eT, size_t i, coot_backend_t backend >
using macro_elem_type = concat_str
  <
  macro_et_def,
  index_to_str<i>,
  equals,
  elem_type_str<eT, backend>
  >;
