// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



class mtop_any
  : public traits_op_xvec
  {
  public:

  template<typename T1>               inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_any>& in);
  // special handling of a conversion linked with an any()
  template<typename T1, typename eT2> inline static void apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_any>& in);
  // special handling of an any() on a relational operation
  // (we just try to apply some simple optimizations to reduce it to a single kernel)
  template<typename T1, typename mtop_type> inline static void apply(Mat<uword>& out,
                                                                     const mtOp<uword, mtOp<uword, T1, mtop_type>, mtop_any>& in,
                                                                     const typename enable_if<is_same_type<mtop_type, mtop_conv_to>::no>::result* junk = 0);

  // for special handling of conversions linked with an any()
  template<typename eT, typename eT2> inline static void apply_direct(Mat<uword>& out, const Mat<eT>& in,     const uword dim);
  template<typename eT, typename eT2> inline static void apply_direct(Mat<uword>& out, const subview<eT>& in, const uword dim);

  template<typename T1> inline static bool any_vec(T1& X);
  // for nested applications
  template<typename out_eT, typename T1> inline static bool any_vec(const mtOp<out_eT, T1, mtop_any>& op);
  // for applications with conversions
  template<typename eT2, typename T1>    inline static bool any_vec(const mtOp<eT2, T1, mtop_conv_to>& op);
  // special handling of an any() on a relational operation
  // (we just try to apply some simple optimizations to reduce it to a single kernel)
  template<typename T1, typename mtop_type> inline static bool any_vec(const mtOp<uword, T1, mtop_type>& in,
                                                                       const typename enable_if<is_same_type<mtop_type, mtop_any>::no>::result* junk1 = 0,
                                                                       const typename enable_if<is_same_type<mtop_type, mtop_conv_to>::no>::result* junk2 = 0);

  template<typename out_eT, typename T1> inline static uword compute_n_rows(const mtOp<out_eT, T1, mtop_any>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename out_eT, typename T1> inline static uword compute_n_cols(const mtOp<out_eT, T1, mtop_any>& op, const uword in_n_rows, const uword in_n_cols);
  };
