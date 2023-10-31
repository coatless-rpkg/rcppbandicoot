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



class op_sum
  : public traits_op_xvec
  {
  public:

  template<typename eT2, typename T1>
  inline static void apply(Mat<eT2>& out, const Op<T1, op_sum>& in);

  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword dim, const bool post_conv_apply);

  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const subview<in_eT>& sv, const uword dim, const bool post_conv_apply);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_sum>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_sum>& op, const uword in_n_rows, const uword in_n_cols);
  };
