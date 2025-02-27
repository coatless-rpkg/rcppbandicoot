// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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



template<typename out_eT, typename T1>
inline
void
mtop_imag::apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_imag>& in)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> U(in.q);

  out.set_size(U.M.n_rows, U.M.n_cols);

  coot_rt_t::extract_cx(out.get_dev_mem(false),
                        0, 0, out.n_rows,
                        U.M.get_dev_mem(false),
                        U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                        U.M.n_rows, U.M.n_cols,
                        true /* extract imaginary elements */);
  }



template<typename out_eT, typename T1>
inline
uword
mtop_imag::compute_n_rows(const mtOp<out_eT, T1, mtop_imag>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_imag::compute_n_cols(const mtOp<out_eT, T1, mtop_imag>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
