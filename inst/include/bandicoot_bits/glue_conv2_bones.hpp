// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



class glue_conv2
  : public traits_glue_default
  {
  public:

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_conv2>& in);

  template<typename out_eT, typename eT>
  inline static void apply_direct(Mat<out_eT>& out, const Mat<eT>& A_in, const Mat<eT>& B_in, const uword mode, const typename enable_if<is_same_type<out_eT, eT>::no>::result* junk = 0);

  template<typename eT>
  inline static void apply_direct(Mat<eT>& out, const Mat<eT>& A_in, const Mat<eT>& B_in, const uword mode);

  // Utilities to compute buffer sizes.
  template<typename eT>
  inline static void get_gemv_full_sizes(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset);

  template<typename eT>
  inline static void get_gemv_same_sizes(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset);

  template<typename eT>
  inline static void get_gemv_same_sizes_small(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset);

  // Utilities to fill the buffer.
  template<typename eT>
  inline static void fill_gemv_buffer_top_bottom(Mat<eT>& buffer, const uword buffer_top_padding, const uword buffer_bottom_padding, const uword kernel_rows);

  template<typename eT>
  inline static void fill_gemv_buffer_col(Mat<eT>& buffer, const uword i, const uword j, const Mat<eT>& A, const Mat<eT>& K, const uword buffer_top_padding, const uword A_col_offset);

  template<typename T1, typename T2> inline static uword compute_n_rows(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };
