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
// `isnan_str` specialization for OpenCL: for OpenCL, we need to augment the
// coot_isnan_ with the element type; e.g., coot_isnan_double().
//



struct isnan_str_cl_prefix
  {
  static inline constexpr size_t len() { return 11; }
  static inline constexpr char_array<12> str() { return char_array<12>{ "coot_isnan_" }; }
  };



template<typename eT>
struct isnan_str< eT, CL_BACKEND > : public concat_str
  <
  isnan_str_cl_prefix,
  elem_type_str<eT, CL_BACKEND>
  > { };
