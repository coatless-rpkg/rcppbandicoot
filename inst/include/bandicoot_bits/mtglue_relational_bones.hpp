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



class mtglue_rel_lt
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_lt>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_lt>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_lt>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_gt
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_gt>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_gt>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_gt>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_lteq
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_lteq>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_lteq>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_lteq>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_gteq
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_gteq>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_gteq>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_gteq>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_eq
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_eq>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_eq>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_eq>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_noteq
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_noteq>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_noteq>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_noteq>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_and
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_and>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_and>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_and>& glue, const uword in_n_rows, const uword in_n_cols);
  };



class mtglue_rel_or
  : public traits_glue_or
  {
  public:

  template<typename T1, typename T2>
  inline static void apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_or>& X);

  template<typename T1, typename T2> inline static uword compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_or>& glue, const uword in_n_rows, const uword in_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_or>& glue, const uword in_n_rows, const uword in_n_cols);
  };
