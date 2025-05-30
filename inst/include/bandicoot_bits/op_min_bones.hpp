// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2021-2025 Ryan Curtin (https://www.ratml.org/)
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



class op_min
  : public traits_op_xvec
  {
  public:

  //
  // for use in delayed operations on matrices
  //

  template<typename eT2, typename T1>
  inline static void apply(Mat<eT2>& out, const Op<T1, op_min>& in);

  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword dim, const bool post_conv_apply);

  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const subview<in_eT>& sv, const uword dim, const bool post_conv_apply);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_min>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_min>& op, const uword in_n_rows, const uword in_n_cols);

  //
  // for use in delayed operations on cubes
  //

  template<typename eT2, typename T1>
  inline static void apply(Cube<eT2>& out, const OpCube<T1, op_min>& in);

  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Cube<out_eT>& out, const Cube<in_eT>& A, const uword dim, const bool post_conv_apply);

  template<typename T1> inline static uword compute_n_rows(const OpCube<T1, op_min>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_cols(const OpCube<T1, op_min>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_slices(const OpCube<T1, op_min>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);

  //
  // for use in direct operations
  //

  template<typename T1>
  inline static typename T1::elem_type apply_direct(const Base<typename T1::elem_type, T1>& in);

  template<typename T1>
  inline static typename T1::elem_type apply_direct(const BaseCube<typename T1::elem_type, T1>& in);
  };
