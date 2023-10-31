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
op_mean::apply(Mat<out_eT>& out, const Op<T1, op_mean>& in)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> U(in.m);

  const uword dim = in.aux_uword_a;
  apply_direct(out, U.M, dim, false);
  }



template<typename eT, typename T1>
inline
void
op_mean::apply(Mat<eT>& out, const Op<mtOp<eT, T1, mtop_conv_to>, op_mean>& in)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> U(in.m.q);

  const uword dim = in.aux_uword_a;
  apply_direct(out, U.M, dim, true);
  }



template<typename out_eT, typename in_eT>
inline
void
op_mean::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  // We can't apply the kernel into an alias.
  copy_alias<in_eT> C(in, out);

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

  coot_rt_t::mean(out.get_dev_mem(false), C.M.get_dev_mem(false),
                  in.n_rows, in.n_cols,
                  dim, post_conv_apply,
                  0, 1,
                  0, 0, C.M.n_rows);
  }



template<typename out_eT, typename in_eT>
inline
void
op_mean::apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  // If `in` is a subview of `out`, we need to make a copy.
  if (((void*) &in.m) == ((void*) &out))
    {
    Mat<in_eT> tmp(in);
    apply_direct(out, tmp, dim, post_conv_apply);
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

  coot_rt_t::mean(out.get_dev_mem(false), in.m.get_dev_mem(false),
                  in.n_rows, in.n_cols,
                  dim, post_conv_apply,
                  0, 1,
                  in.aux_row1, in.aux_col1, in.m.n_rows);
  }



template<typename T1>
inline
typename T1::elem_type
op_mean::mean_all(const T1& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X);

  if (U.M.n_elem == 0)
    {
    return eT(0);
    }

  return mean_all_direct(U.M);
  }



template<typename eT>
inline
eT
op_mean::mean_all_direct(const Mat<eT>& M)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::accu(M.get_dev_mem(false), M.n_elem) / eT(M.n_elem);
  }



template<typename eT>
inline
eT
op_mean::mean_all_direct(const subview<eT>& M)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::accu_subview(M.m.get_dev_mem(false), M.m.n_rows, M.aux_row1, M.aux_col1, M.n_rows, M.n_cols) / eT(M.n_elem);
  }



template<typename T1>
inline
uword
op_mean::compute_n_rows(const Op<T1, op_mean>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_a;
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
op_mean::compute_n_cols(const Op<T1, op_mean>& op, const uword in_n_rows, const uword in_n_cols)
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
