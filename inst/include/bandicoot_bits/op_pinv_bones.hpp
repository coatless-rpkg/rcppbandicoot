// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



class op_pinv
  : public traits_op_passthru
  {
  public:

  //
  // for use in delayed operations
  //

  template<typename eT2, typename T1>
  inline static void apply(Mat<eT2>& out, const Op<T1, op_pinv>& in);

  template<typename eT2, typename T1>
  inline static std::tuple<bool, std::string> apply_direct(Mat<eT2>& out, const T1& in, const typename T1::elem_type tol);

  // apply to a diagonal matrix that is represented as a vector
  template<typename eT>
  inline static std::tuple<bool, std::string> apply_direct_diag(Mat<eT>& out, const Mat<eT>& in, const eT tol);

  template<typename eT2, typename eT1>
  inline static std::tuple<bool, std::string> apply_direct_diag(Mat<eT2>& out, const Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk = 0);

  // apply to a symmetric matrix using eigendecomposition (faster than SVD)
  template<typename eT>
  inline static std::tuple<bool, std::string> apply_direct_sym(Mat<eT>& out, Mat<eT>& in, const eT tol);

  template<typename eT2, typename eT1>
  inline static std::tuple<bool, std::string> apply_direct_sym(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk = 0);

  // apply to a general matrix using SVD (slowest)
  template<typename eT>
  inline static std::tuple<bool, std::string> apply_direct_gen(Mat<eT>& out, Mat<eT>& in, const eT tol);

  template<typename eT2, typename eT1>
  inline static std::tuple<bool, std::string> apply_direct_gen(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk = 0);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols);
  };
