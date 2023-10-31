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



class mtop_find_nan
  : public traits_op_col
  {
  public:

  template<typename T1> inline static void apply(Mat<uword>& out, const mtOp<uword, T1, mtop_find_nan>& in);

  template<typename out_eT, typename T1> inline static uword compute_n_rows(const mtOp<out_eT, T1, mtop_find_nan>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename out_eT, typename T1> inline static uword compute_n_cols(const mtOp<out_eT, T1, mtop_find_nan>& op, const uword in_n_rows, const uword in_n_cols);
  };



// Special overload to hold computed result, since we cannot know the size without actually computing the result.
// This is used to store the result.
template<typename T1>
struct Op_traits<T1, mtop_find_nan, true>
  {
  static constexpr bool is_row  = mtop_find_nan::template traits<T1>::is_row;
  static constexpr bool is_col  = mtop_find_nan::template traits<T1>::is_col;
  static constexpr bool is_xvec = mtop_find_nan::template traits<T1>::is_xvec;

  Mat<uword> computed_result;
  bool is_computed;

  Op_traits() : is_computed(false) { }
  };
