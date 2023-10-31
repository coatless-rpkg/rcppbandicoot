// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2023 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2023 National ICT Australia (NICTA)
// Copyright 2021-2023 Marcus Edel (http://kurg.org)
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_min::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_min>& X)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> UA(X.A);
  no_conv_unwrap<T2> UB(X.B);

  coot_debug_assert_same_size(UA.M.n_rows, UA.M.n_cols, UB.M.n_rows, UB.M.n_cols, "element-wise min()");

  out.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_min_array,
                     out.get_dev_mem(false), UA.get_dev_mem(false), UB.get_dev_mem(false),
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     UA.get_row_offset(), UA.get_col_offset(), UA.get_M_n_rows(),
                     UB.get_row_offset(), UB.get_col_offset(), UB.get_M_n_rows());
  }



template<typename T1, typename T2>
inline
uword
glue_min::compute_n_rows(const Glue<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
glue_min::compute_n_cols(const Glue<T1, T2, glue_min>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }
