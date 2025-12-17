// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (https://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



struct mtglue_mixed_times
  {
  template<typename T1, typename T2>
  struct traits
    {
    static constexpr bool is_row  = T1::is_row;
    static constexpr bool is_col  = T2::is_col;
    static constexpr bool is_xvec = false;
    };

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& X);

  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };



struct mtglue_mixed_plus
  : public traits_glue_or
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& X);

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& X);

  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };



struct mtglue_mixed_minus
  : public traits_glue_or
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& X);

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& X);

  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };



struct mtglue_mixed_div
  : public traits_glue_or
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& X);

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& X);

  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };



struct mtglue_mixed_schur
  : public traits_glue_or
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& X);

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& X);

  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);

  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword   compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  template<typename out_eT, typename T1, typename T2> inline static uword compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices);
  };
