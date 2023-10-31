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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_cross::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_cross>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> UA(in.A);
  const unwrap<T2> UB(in.B);

  coot_debug_check( (UA.M.n_elem != 3 || UB.M.n_elem != 3), "cross(): each vector must have 3 elements" );

  out.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::cross(out.get_dev_mem(false), UA.M.get_dev_mem(false), UB.M.get_dev_mem(false));
  }



template<typename T1, typename T2>
inline
uword
glue_cross::compute_n_rows(const Glue<T1, T2, glue_cross>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
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
glue_cross::compute_n_cols(const Glue<T1, T2, glue_cross>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }
