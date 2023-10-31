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



class mtop_rel_lt_pre
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lt_pre>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lt_pre>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_lt_pre>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_lt_pre>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_lt_post
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lt_post>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lt_post>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_lt_post>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_lt_post>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_gt_pre
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gt_pre>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gt_pre>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_gt_pre>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_gt_pre>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_gt_post
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gt_post>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gt_post>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_gt_post>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_gt_post>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_lteq_pre
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lteq_pre>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lteq_pre>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_lteq_pre>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_lteq_pre>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_lteq_post
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lteq_post> &X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lteq_post>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_lteq_post>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_lteq_post>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_gteq_pre
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gteq_pre>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gteq_pre>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_gteq_pre>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_gteq_pre>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_gteq_post
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gteq_post>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gteq_post>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_gteq_post>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_gteq_post>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_eq
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_eq>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_eq>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_eq>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_eq>& op, const uword in_n_rows, const uword in_n_cols);
  };



class mtop_rel_noteq
  : public traits_op_passthru
  {
  public:

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_noteq>& X);

  // specialization for conversions, to avoid computing the conversion
  template<typename eT2, typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_noteq>& X);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_rel_noteq>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_rel_noteq>& op, const uword in_n_rows, const uword in_n_cols);
  };
