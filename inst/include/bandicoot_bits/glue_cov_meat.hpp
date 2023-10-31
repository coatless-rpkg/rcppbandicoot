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
glue_cov::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_cov>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  // We use the same unwrapping strategy as cor() here (that is, if we get a row
  // vector, we treat is as a column vector instead).
  const special_cor_cov_unwrap<T1> U1(in.A);
  const special_cor_cov_unwrap<T2> U2(in.B);

  const uword AA_n_rows = U1.get_n_rows();
  const uword AA_n_cols = U1.get_n_cols();
  const uword BB_n_rows = U2.get_n_rows();
  const uword BB_n_cols = U2.get_n_cols();

  coot_debug_assert_mul_size(AA_n_cols, AA_n_rows, BB_n_rows, BB_n_cols, "cov()");

  if (U1.M.n_elem == 0 || U2.M.n_elem == 0)
    {
    out.reset();
    return;
    }

  const uword N         = AA_n_rows;
  const uword norm_type = in.aux_uword;
  const eT norm_val     = (norm_type == 0) ? ( (N > 1) ? eT(N - 1) : eT(1) ) : eT(N);

  // TODO: a dedicated kernel for this particular operation would be widely useful
  Row<eT> mean_vals_AA(AA_n_cols);
  coot_rt_t::mean(mean_vals_AA.get_dev_mem(false), U1.get_dev_mem(false),
                  AA_n_rows, AA_n_cols,
                  0, true,
                  0, 1,
                  U1.get_row_offset(), U1.get_col_offset(), U1.get_M_n_rows());

  Row<eT> mean_vals_BB(BB_n_cols);
  coot_rt_t::mean(mean_vals_BB.get_dev_mem(false), U2.get_dev_mem(false),
                  BB_n_rows, BB_n_cols,
                  0, true,
                  0, 1,
                  U2.get_row_offset(), U2.get_col_offset(), U2.get_M_n_rows());

  Mat<eT> tmp_AA;
  if (U1.use_local_mat)
    {
    tmp_AA = U1.local_mat;
    }
  else
    {
    tmp_AA = U1.M;
    }

  Mat<eT> tmp_BB;
  if (U2.use_local_mat)
    {
    tmp_BB = U2.local_mat;
    }
  else
    {
    tmp_BB = U2.M;
    }

  for (uword i = 0; i < tmp_AA.n_rows; ++i)
    {
    tmp_AA.row(i) -= mean_vals_AA;
    }
  for (uword i = 0; i < tmp_BB.n_rows; ++i)
    {
    tmp_BB.row(i) -= mean_vals_BB;
    }

  out = conv_to<Mat<out_eT>>::from((tmp_AA.t() * tmp_BB) / norm_val);
  }



template<typename T1, typename T2>
inline
uword
glue_cov::compute_n_rows(const Glue<T1, T2, glue_cov>& op, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(op);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  if (A_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead.
    return 1;
    }
  else
    {
    return A_n_cols;
    }
  }



template<typename T1, typename T2>
inline
uword
glue_cov::compute_n_cols(const Glue<T1, T2, glue_cov>& op, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(op);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);

  if (B_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead.
    return 1;
    }
  else
    {
    return B_n_cols;
    }
  }
