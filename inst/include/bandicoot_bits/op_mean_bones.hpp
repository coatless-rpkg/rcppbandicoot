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


class op_mean
  : public traits_op_xvec
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1, op_mean>& in);
  template<typename eT, typename T1>     inline static void apply(Mat<eT>& out,     const Op<mtOp<eT, T1, mtop_conv_to>, op_mean>& in);

  template<typename out_eT, typename in_eT> inline static void apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const uword dim, const bool post_conv_apply);
  template<typename out_eT, typename in_eT> inline static void apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const uword dim, const bool post_conv_apply);

  template<typename T1> inline static typename T1::elem_type mean_all(const T1& X);

  template<typename eT> inline static eT mean_all_direct(const Mat<eT>& M);
  template<typename eT> inline static eT mean_all_direct(const subview<eT>& M);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_mean>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_mean>& op, const uword in_n_rows, const uword in_n_cols);
  };
