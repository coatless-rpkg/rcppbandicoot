// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



template<typename mtglue_type>
class mtglue_rel_core : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type> >& X);
  template<typename T1, typename T2>
  inline static void apply(Cube<uword>& out, const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type> >& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename T1, typename T2> inline static uword compute_n_slices(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };



class mtglue_rel_lt
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "L";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_lt"; } };
  struct func_body { static inline constexpr auto& str() { return "x < y";   } };

  // inline out_eT coot_lt(const eT x, const eT y) { return x < y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_gt
  : public traits_glue_or
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "G";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_gt"; } };
  struct func_body { static inline constexpr auto& str() { return "x > y";   } };

  // inline out_eT coot_gt(const eT x, const eT y) { return x > y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_lteq
  : public traits_glue_or
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "l";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_le"; } };
  struct func_body { static inline constexpr auto& str() { return "x <= y";  } };

  // inline out_eT coot_le(const eT x, const eT y) { return x <= y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_gteq
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "g";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_ge"; } };
  struct func_body { static inline constexpr auto& str() { return "x >= y";  } };

  // inline out_eT coot_ge(const eT x, const eT y) { return x >= y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_eq
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "e";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_eq"; } };
  struct func_body { static inline constexpr auto& str() { return "x == y";  } };

  // inline out_eT coot_eq(const eT x, const eT y) { return x == y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_noteq
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "n";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_neq"; } };
  struct func_body { static inline constexpr auto& str() { return "x != y";   } };

  // inline out_eT coot_neq(const eT x, const eT y) { return x != y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_and
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "A";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_and"; } };
  struct func_body { static inline constexpr auto& str() { return "x && y";   } };

  // inline out_eT coot_and(const eT x, const eT y) { return x && y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };



class mtglue_rel_or
  {
  public:

  struct prefix    { static inline constexpr auto& str() { return "O";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_or"; } };
  struct func_body { static inline constexpr auto& str() { return "x || y";  } };

  // inline out_eT coot_or(const eT x, const eT y) { return x || y; }
  template<typename out_eT, typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::mtglue_inline_function< out_eT, eT, backend, func_name, func_body > { };
  };
