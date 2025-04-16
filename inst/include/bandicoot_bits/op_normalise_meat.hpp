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



template<typename T1>
inline
void
op_normalise_vec::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_normalise_vec>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const uword p = in.aux_uword_a;

  coot_debug_check( (p == 0), "normalise(): parameter 'p' must be greater than zero" );

  const unwrap<T1> U(in.m);

  const eT norm_val_a = norm(U.M, p);
  const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

  if (U.is_alias(out) && norm_val_b == eT(1))
    {
    // Shortcut: if the norm is 1, it's already normalised.
    return;
    }
  else if (U.is_alias(out))
    {
    out /= norm_val_b;
    }
  else if (norm_val_b == eT(1))
    {
    // Shortcut: if the norm is 1, it's already normalised.
    out = U.M;
    }
  else
    {
    out = U.M / norm_val_b;
    }
  }



template<typename eT, typename T1>
inline
void
op_normalise_vec::apply(Mat<eT>& out, const Op<mtOp<eT, T1, mtop_conv_to>, op_normalise_vec>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT2;

  const uword p = in.aux_uword_a;

  coot_debug_check( (p == 0), "normalise(): parameter 'p' must be greater than zero" );

  const unwrap<T1> U(in.m.q);

  const eT norm_val_a = norm(in.m, p);
  const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

  out.set_size(U.M.n_rows, U.M.n_cols);

  if (norm_val_b == eT2(1))
    {
    // Shortcut: if the norm is 1, it's already normalized.
    coot_rt_t::copy_mat(out.get_dev_mem(false), U.get_dev_mem(false),
                        out.n_rows, out.n_cols,
                        0, 0, out.n_rows,
                        U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  else
    {
    // Note that normalisation happens *after* conversion.
    coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                          out.get_dev_mem(false), U.get_dev_mem(false),
                          eT2(1), eT(norm_val_b),
                          out.n_rows, out.n_cols, 1,
                          0, 0, 0, out.n_rows, out.n_cols,
                          U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), out.n_cols /* ignored */);
    }
  }



template<typename T1>
inline
uword
op_normalise_vec::compute_n_rows(const Op<T1, op_normalise_vec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_normalise_vec::compute_n_cols(const Op<T1, op_normalise_vec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
op_normalise_mat::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_normalise_mat>& in)
  {
  coot_extra_debug_sigprint();

  const uword p   = in.aux_uword_a;
  const uword dim = in.aux_uword_b;

  coot_debug_check( (p   == 0), "normalise(): parameter 'p' must be greater than zero" );
  coot_debug_check( (dim >  1), "normalise(): parameter 'dim' must be 0 or 1"          );

  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> S(U.M);

  if (((void*) &out) == ((void*) &S.M))
    {
    op_normalise_mat::apply_alias(out, p, dim);
    }
  else
    {
    op_normalise_mat::apply_direct(out, S.M, p, dim);
    }
  }



template<typename eT>
inline
void
op_normalise_mat::apply_alias(Mat<eT>& A, const uword p, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (A.n_elem == 0)
    {
    return;
    }

  if (dim == 0)
    {
    const uword n_cols = A.n_cols;

    for (uword c = 0; c < n_cols; ++c)
      {
      const eT norm_val_a = norm(A.col(c), p);
      const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

      if (norm_val_b != eT(1))
        {
        A.col(c) /= norm_val_b;
        }
      }
    }
  else
    {
    const uword n_rows = A.n_rows;

    for (uword r = 0; r < n_rows; ++r)
      {
      const eT norm_val_a = norm(A.row(r), p);
      const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

      if (norm_val_b != eT(1))
        {
        A.row(r) /= norm_val_b;
        }
      }
    }
  }



template<typename eT>
inline
void
op_normalise_mat::apply_direct(Mat<eT>& out, const Mat<eT>& A, const uword p, const uword dim)
  {
  coot_extra_debug_sigprint();

  out.set_size(A.n_rows, A.n_cols);

  if (A.n_elem == 0)
    {
    return;
    }

  if (dim == 0)
    {
    const uword n_cols = A.n_cols;

    for (uword c = 0; c < n_cols; ++c)
      {
      const eT norm_val_a = norm(A.col(c), p);
      const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

      if (norm_val_b != eT(1))
        {
        // TODO: this (and similar operations) could be painful since they may involve subview extractions.
        out.col(c) = A.col(c) / norm_val_b;
        }
      else
        {
        out.col(c) = A.col(c);
        }
      }
    }
  else
    {
    const uword n_rows = A.n_rows;

    for (uword r = 0; r < n_rows; ++r)
      {
      const eT norm_val_a = norm(A.row(r), p);
      const eT norm_val_b = (norm_val_a != eT(0)) ? norm_val_a : eT(1);

      if (norm_val_b != eT(1))
        {
        out.row(r) = A.row(r) / norm_val_b;
        }
      else
        {
        out.row(r) = A.row(r);
        }
      }
    }
  }



template<typename T1>
inline
uword
op_normalise_mat::compute_n_rows(const Op<T1, op_normalise_mat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_normalise_mat::compute_n_cols(const Op<T1, op_normalise_mat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
