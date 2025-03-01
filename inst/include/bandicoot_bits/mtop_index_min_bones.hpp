// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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



class mtop_index_min
  : public traits_op_xvec
  {
  public:

  //
  // for use in delayed operations
  //

  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_index_min>& in);

  template<typename eT>
  inline static void apply_noalias(Mat<uword>& out, const Mat<eT>& A, const uword dim);

  template<typename eT>
  inline static void apply_noalias(Mat<uword>& out, const subview<eT>& sv, const uword dim);

  template<typename T1> inline static uword compute_n_rows(const mtOp<uword, T1, mtop_index_min>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const mtOp<uword, T1, mtop_index_min>& op, const uword in_n_rows, const uword in_n_cols);

  //
  // for use in direct operations
  //

  template<typename T1>
  inline static uword apply_direct(const T1& in);
  };
