// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2021-2022 Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



class op_vectorise_col
  : public traits_op_col
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_col>& in);

  // output_is_row is only used by op_vectorise_all in special situations
  template<typename eT> inline static void apply_direct(Mat<eT>& out, const     Mat<eT>& in, const bool output_is_row = false);
  template<typename eT> inline static void apply_direct(Mat<eT>& out, const subview<eT>& in, const bool output_is_row = false);

  template<typename out_eT, typename in_eT> inline static void apply_direct(Mat<out_eT>& out, const     Mat<in_eT>& in, const bool output_is_row = false);
  template<typename out_eT, typename in_eT> inline static void apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const bool output_is_row = false);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols);
  };



class op_vectorise_all
  : public traits_op_xvec
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_all>& in);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols);
  };



class op_vectorise_row
  : public traits_op_row
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_row>& in);

  template<typename T1> inline static void apply_direct(Mat<typename T1::elem_type>& out, const T1& in);

  template<typename out_eT, typename T1> inline static void apply_direct(Mat<out_eT>& out, const T1& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* = 0);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols);
  };



class op_vectorise_cube_col
  : public traits_op_col
  {
  public:
  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const CubeToMatOp<T1, op_vectorise_cube_col>& in);

  template<typename T1> inline static uword compute_n_rows(const CubeToMatOp<T1, op_vectorise_cube_col>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_cols(const CubeToMatOp<T1, op_vectorise_cube_col>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  };

