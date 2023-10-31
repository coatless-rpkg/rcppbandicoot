// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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



class op_repmat
  : public traits_op_default
  {
  public:

  template<typename out_eT, typename T1> inline static void apply_noalias(Mat<out_eT>& out, const T1& X, const uword copies_per_row, const uword copies_per_col);

  // repmat() with optional combined conversion
  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1, op_repmat>& in);

  // Catch conversions directly before a repmat (we can combine these with the repmat operation).
  // (The other way around, with conversions directly after a repmat, is caught by mtOp handling.)
  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to>, op_repmat>& in);

  template<typename T1> static inline uword compute_n_rows(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> static inline uword compute_n_cols(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols);
  };
