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



class op_normalise_vec
  : public traits_op_passthru
  {
  public:

  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_normalise_vec>& in);
  template<typename eT, typename T1> inline static void apply(Mat<eT>& out, const Op<mtOp<eT, T1, mtop_conv_to>, op_normalise_vec>& in);

  template<typename T1> static inline uword compute_n_rows(const Op<T1, op_normalise_vec>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> static inline uword compute_n_cols(const Op<T1, op_normalise_vec>& op, const uword in_n_rows, const uword in_n_cols);
  };



class op_normalise_mat
  : public traits_op_default
  {
  public:

  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_normalise_mat>& in);

  template<typename eT> inline static void apply_alias(Mat<eT>& A, const uword p, const uword dim);
  template<typename eT> inline static void apply_direct(Mat<eT>& out, const Mat<eT>& A, const uword p, const uword dim);

  template<typename T1> static inline uword compute_n_rows(const Op<T1, op_normalise_mat>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> static inline uword compute_n_cols(const Op<T1, op_normalise_mat>& op, const uword in_n_rows, const uword in_n_cols);
  };
