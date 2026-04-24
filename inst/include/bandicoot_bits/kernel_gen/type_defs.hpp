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
// template metaprogramming utilities to provide compile-time definitions for each element type
//



//
// For each backend and supported type, this must be defined with the function
//
// static inline constexpr auto& src_str() { return
//    #include "defs/<type>_defs.<ext>"
//    ; }
//
// (Note this particular spacing is needed because the #include must be on its own line.)
//
// This struct is specialized accordingly in cuda/type_defs.hpp and opencl/type_defs.hpp.
//

template<coot_backend_t backend, typename eT>
struct type_def { };



//
// Wrapper class for `type_def`, used by `type_defs` to actually provide the strings of
// the sources defined in `type_def<>` specializations.
//

template<coot_backend_t backend, typename eT>
struct type_def_str : public type_def<backend, eT>
  {
  static inline constexpr size_t len() { return sizeof(type_def<backend, eT>::src_str()) - 1; }
  static inline constexpr char_array<len() + 1> str() { return to_array::array(type_def<backend, eT>::src_str()); }
  };



//
// Helper struct to get the length of the definitions needed for all the types in a list of types.
//

template<typename T>
struct type_defs_len_helper { };

template<typename T, typename... Ts>
struct type_defs_len_helper< std::tuple<T, Ts...> >
  {
  static inline constexpr size_t len()
    {
    return T::len() + type_defs_len_helper< std::tuple<Ts...> >::len();
    }
  };

template<>
struct type_defs_len_helper< std::tuple<> >
  {
  static inline constexpr size_t len()
    {
    return 0;
    }
  };



//
// Helper struct to concatenate the sources required for all the types in a list of types.
//

template<typename T>
struct type_defs_concat_helper { };

template<typename T, typename... Ts>
struct type_defs_concat_helper< std::tuple<T, Ts...> >
  {
  static inline size_t concat_src(std::string& result, const size_t offset)
    {
    result.replace(offset, T::len(), T::src_str());
    return type_defs_concat_helper< std::tuple<Ts...> >::concat_src(result, offset + T::len());
    }
  };

template<>
struct type_defs_concat_helper< std::tuple<> >
  {
  static inline size_t concat_src(std::string&, const size_t offset) { return offset; }
  };



//
// Holder object for multiple `type_def`s for different types.
//

template<coot_backend_t backend, typename T>
struct type_defs { };

template<coot_backend_t backend, typename... eTs>
struct type_defs<backend, std::tuple< eTs... > >
  {
  typedef typename std::tuple< type_def_str<backend, eTs>... > result;
  };
