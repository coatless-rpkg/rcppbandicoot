// SPDX-License-Identifier: Apache-2.0
// 
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



template<typename out_eT, typename T1>
inline
void
op_var::apply(Mat<out_eT>& out, const Op<T1, op_var>& in)
  {
  coot_extra_debug_sigprint();

  const uword norm_type = in.aux_uword_a;
  const uword dim = in.aux_uword_b;

  unwrap<T1> U(in.m);

  apply_direct(out, U.M, dim, norm_type);
  }



template<typename out_eT, typename in_eT>
inline
void
op_var::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // We require a temporary for the output because we don't have any kernels that perform a conversion and variance computation at the same time.
  Mat<in_eT> tmp;
  apply_direct(tmp, in, dim, norm_type);
  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy_mat(out.get_dev_mem(false), tmp.get_dev_mem(false),
                      out.n_rows, out.n_cols,
                      0, 0, out.n_rows,
                      0, 0, tmp.n_rows);
  }



template<typename eT>
inline
void
op_var::apply_direct(Mat<eT>& out, const Mat<eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // Our kernel can't handle aliases.
  copy_alias<eT> C(in, out);

  if (dim == 0)
    {
    out.set_size(in.n_rows > 0 ? 1 : 0, in.n_cols);
    }
  else
    {
    out.set_size(in.n_rows, in.n_cols > 0 ? 1 : 0);
    }

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // First we need to compute the mean.
  Mat<eT> mean;
  op_mean::apply_direct(mean, C.M, dim, false);

  coot_rt_t::var(out.get_dev_mem(false), C.M.get_dev_mem(false), mean.get_dev_mem(false),
                 in.n_rows, in.n_cols,
                 dim, norm_type,
                 0, 1,
                 0, 0, C.M.n_rows,
                 0, 1);
  }



template<typename out_eT, typename in_eT>
inline
void
op_var::apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // This will require a conversion, so just extract the subview.
  Mat<out_eT> tmp(in);
  apply_direct(out, tmp, dim, norm_type);
  }



template<typename eT>
inline
void
op_var::apply_direct(Mat<eT>& out, const subview<eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // If `in` is a subview of `out`, we need to extract it.
  if (((void*) &in.m) == ((void*) &out))
    {
    Mat<eT> tmp(in);
    apply_direct(out, tmp, dim, norm_type);
    return;
    }

  if (dim == 0)
    {
    out.set_size(in.n_rows > 0 ? 1 : 0, in.n_cols);
    }
  else
    {
    out.set_size(in.n_rows, in.n_cols > 0 ? 1 : 0);
    }

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // First, compute the mean.
  Mat<eT> mean;
  op_mean::apply_direct(mean, in, dim, false);

  coot_rt_t::var(out.get_dev_mem(false), in.m.get_dev_mem(false), mean.get_dev_mem(false),
                 in.n_rows, in.n_cols,
                 dim, norm_type,
                 0, 1,
                 in.aux_row1, in.aux_col1, in.m.n_rows,
                 0, 1);
  }



template<typename T1>
inline
typename T1::elem_type
op_var::var_vec(const T1& X, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X);
  if (U.M.n_elem == 0)
    {
    return eT(0);
    }

  const eT mean_val = op_mean::mean_all_direct(U.M);
  return var_vec_direct(U.M, mean_val, norm_type);
  }



template<typename eT>
inline
eT
op_var::var_vec_direct(const Mat<eT>& M, const eT mean_val, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::var_vec(M.get_dev_mem(false), mean_val, M.n_elem, norm_type);
  }



template<typename eT>
inline
eT
op_var::var_vec_direct(const subview<eT>& M, const eT mean_val, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::var_vec_subview(M.m.get_dev_mem(false), mean_val, M.m.n_rows, M.m.n_cols, M.aux_row1, M.aux_col1, M.n_rows, M.n_cols, norm_type);
  }



template<typename T1>
inline
uword
op_var::compute_n_rows(const Op<T1, op_var>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return std::min(in_n_rows, uword(1)); // either 0 or 1
    }
  else
    {
    return in_n_rows;
    }
  }



template<typename T1>
inline
uword
op_var::compute_n_cols(const Op<T1, op_var>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return in_n_cols;
    }
  else
    {
    return std::min(in_n_cols, uword(1)); // either 0 or 1
    }
  }
