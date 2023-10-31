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



template<typename out_eT, typename T1>
inline
void
op_cov::apply(Mat<out_eT>& out, const Op<T1, op_cov>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  // If the input is a row vector, we treat it as a column vector instead.
  const special_cor_cov_unwrap<T1> U(in.m);

  if (U.M.n_elem == 0)
    {
    out.reset();
    return;
    }

  const uword AA_n_rows = U.get_n_rows();
  const uword AA_n_cols = U.get_n_cols();

  const uword N         = AA_n_rows;
  const uword norm_type = in.aux_uword_a;
  const eT norm_val     = (norm_type == 0) ? ( (N > 1) ? eT(N - 1) : eT(1) ) : eT(N);

  // TODO: a dedicated kernel for this particular operation would be widely useful
  Row<eT> mean_vals(AA_n_cols);
  coot_rt_t::mean(mean_vals.get_dev_mem(false), U.get_dev_mem(false),
                  AA_n_rows, AA_n_cols,
                  0, true,
                  0, 1,
                  U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  Mat<eT> tmp(AA_n_rows, AA_n_cols);
  coot_rt_t::copy_mat(tmp.get_dev_mem(false), U.get_dev_mem(false),
                      AA_n_rows, AA_n_cols,
                      0, 0, tmp.n_rows,
                      U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());

  for (uword i = 0; i < tmp.n_rows; ++i)
    {
    tmp.row(i) -= mean_vals;
    }

  out = conv_to<Mat<out_eT>>::from((tmp.t() * tmp) / norm_val);
  }



template<typename T1>
inline
uword
op_cov::compute_n_rows(const Op<T1, op_cov>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead, giving a 1x1 covariance matrix.
    return 1;
    }
  else
    {
    return in_n_cols;
    }
  }



template<typename T1>
inline
uword
op_cov::compute_n_cols(const Op<T1, op_cov>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead, giving a 1x1 covariance matrix.
    return 1;
    }
  else
    {
    return in_n_cols;
    }
  }
