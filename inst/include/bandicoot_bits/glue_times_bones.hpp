// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename glue_type, typename T1>
struct depth_lhs
  {
  static const uword num = 0;
  };

template<typename glue_type, typename T1, typename T2>
struct depth_lhs< glue_type, Glue<T1, T2, glue_type> >
  {
  static const uword num = 1 + depth_lhs<glue_type, T1>::num;
  };



template<uword N>
struct glue_times_redirect
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X);
  };


template<>
struct glue_times_redirect<2>
  {
  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X);
  };


template<>
struct glue_times_redirect<3>
  {
  template<typename out_eT, typename T1, typename T2, typename T3>
  inline static void apply(Mat<out_eT>& out, const Glue<Glue<T1, T2, glue_times>, T3, glue_times>& X);
  };


template<>
struct glue_times_redirect<4>
  {
  template<typename out_eT, typename T1, typename T2, typename T3, typename T4>
  inline static void apply(Mat<out_eT>& out, const Glue<Glue<Glue<T1, T2, glue_times>, T3, glue_times>, T4, glue_times>& X);
  };



class glue_times
  {
  public:


  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X);

  //

  template<typename eT1, typename eT2, const bool do_trans_A, const bool do_trans_B>
  inline static uword mul_storage_cost(const Mat<eT1>& A, const Mat<eT2>& B);

  template<typename out_eT, typename T1, typename T2, const bool do_trans_A, const bool do_trans_B, const bool do_scalar_times>
  inline static void apply(Mat<out_eT>& out, const T1& A, const T2& B, const out_eT val);

  template<typename out_eT, typename T1, typename T2, typename T3, const bool do_trans_A, const bool do_trans_B, const bool do_trans_C, const bool do_scalar_times>
  inline static void apply(Mat<out_eT>& out, const T1& A, const T2& B, const T3& C, const out_eT val);

  template<typename out_eT, typename T1, typename T2, typename T3, typename T4, const bool do_trans_A, const bool do_trans_B, const bool do_trans_C, const bool do_trans_D, const bool do_scalar_times>
  inline static void apply(Mat<out_eT>& out, const T1& A, const T2& B, const T3& C, const T4& D, const out_eT val);

  // used by SizeProxy

  template<typename T1, typename T2>
  inline static uword compute_n_rows(const Glue<T1, T2, glue_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2>
  inline static uword compute_n_cols(const Glue<T1, T2, glue_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };



class glue_times_diag
  {
  public:

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times_diag>& X);

  // used by SizeProxy

  template<typename T1, typename T2>
  inline static uword compute_n_rows(const Glue<T1, T2, glue_times_diag>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2>
  inline static uword compute_n_cols(const Glue<T1, T2, glue_times_diag>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };
