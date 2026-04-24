// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



template<typename eglue_type>
struct eglue_core
  {

  //
  // default implementation: use a Proxy and the copy skeleton kernel
  //

  template<typename eT, typename T1, typename T2> inline static void apply(Mat<eT>& out,     const eGlue<T1, T2, eglue_type>& X    );
  template<typename eT, typename T1, typename T2> inline static void apply(Cube<eT>& out,    const eGlueCube<T1, T2, eglue_type>& X);
  };



template<size_t i> struct eglue_arg_names { };
template<>         struct eglue_arg_names<0> { static inline constexpr auto& str() { return "y"; } };



class eglue_plus : public eglue_core<eglue_plus>
  {
  public:

  inline static const char* text() { return "addition"; }

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "p";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_plus"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eglue_minus : public eglue_core<eglue_minus>
  {
  public:

  inline static const char* text() { return "subtraction"; }

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "m";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_minus"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eglue_div : public eglue_core<eglue_div>
  {
  public:

  inline static const char* text() { return "element-wise division"; }

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "d";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_div"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eglue_schur : public eglue_core<eglue_schur>
  {
  public:

  inline static const char* text() { return "element-wise multiplication"; }

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "s";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_times"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eglue_atan2 : public eglue_core<eglue_atan2>
  {
  public:

  inline static const char* text() { return "element-wise atan2"; }

  // inline eT coot_atan2(const eT x, const eT y) { return atan2(x, y); }
  struct prefix    { static inline constexpr auto& str() { return "at2";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_atan2";  } };
  struct func_body { static inline constexpr auto& str() { return "atan2(x, y)"; } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, 1, eglue_arg_names, func_body > { };
  };



class eglue_hypot : public eglue_core<eglue_hypot>
  {
  public:

  inline static const char* text() { return "element-wise hypot"; }

  // inline eT coot_hypot(const eT x, const eT y) { return hypot(x, y); }
  struct prefix    { static inline constexpr auto& str() { return "hyp";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_hypot";  } };
  struct func_body { static inline constexpr auto& str() { return "hypot(x, y)"; } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, 1, eglue_arg_names, func_body > { };
  };



class eglue_max : public eglue_core<eglue_max>
  {
  public:

  inline static const char* text() { return "element-wise maximum"; }

  // we don't need an auxiliary function since we depend on the existing coot_max() definitions for each type
  struct prefix    { static inline constexpr auto& str() { return "cM";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_max";  } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eglue_min : public eglue_core<eglue_min>
  {
  public:

  inline static const char* text() { return "element-wise minimum"; }

  // we don't need an auxiliary function since we depend on the existing coot_max() definitions for each type
  struct prefix    { static inline constexpr auto& str() { return "cm";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_min";  } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };
