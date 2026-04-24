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



template<typename mtop_type>
class mtop_rel_core
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_core<mtop_type> >& X);
  template<typename T1>
  inline static void apply(Cube<uword>& out, const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols);

  template<typename T1> inline static uword compute_n_rows(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_cols(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_slices(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  };



class mtop_rel_lt_pre
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "LP"; } };

  inline static const char* text() { return "operator<"; }

  using equiv_mtglue = mtglue_rel_gt;
  };



class mtop_rel_lt_post
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "Lp"; } };

  inline static const char* text() { return "operator<"; }

  using equiv_mtglue = mtglue_rel_lt;
  };



class mtop_rel_gt_pre
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "GP"; } };

  inline static const char* text() { return "operator>"; }

  using equiv_mtglue = mtglue_rel_lt;
  };



class mtop_rel_gt_post
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "Gp"; } };

  inline static const char* text() { return "operator>"; }

  using equiv_mtglue = mtglue_rel_gt;
  };



class mtop_rel_lteq_pre
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "lP"; } };

  inline static const char* text() { return "operator<="; }

  using equiv_mtglue = mtglue_rel_gteq;
  };



class mtop_rel_lteq_post
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "lp"; } };

  inline static const char* text() { return "operator<="; }

  using equiv_mtglue = mtglue_rel_lteq;
  };



class mtop_rel_gteq_pre
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "gP"; } };

  inline static const char* text() { return "operator>="; }

  using equiv_mtglue = mtglue_rel_lteq;
  };



class mtop_rel_gteq_post
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "gp"; } };

  inline static const char* text() { return "operator>="; }

  using equiv_mtglue = mtglue_rel_gteq;
  };



class mtop_rel_eq
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "e"; } };

  inline static const char* text() { return "operator=="; }

  using equiv_mtglue = mtglue_rel_eq;
  };



class mtop_rel_noteq
  {
  public:

  struct prefix { static inline constexpr auto& str() { return "n"; } };

  inline static const char* text() { return "operator!="; }

  using equiv_mtglue = mtglue_rel_noteq;
  };
