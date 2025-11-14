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


template<typename out_eT, typename T1>
inline
void
op_resize::apply(Mat<out_eT>& out, const Op<T1, op_resize>& in)
  {
  coot_extra_debug_sigprint();

  const uword new_n_rows = in.aux_uword_a;
  const uword new_n_cols = in.aux_uword_b;

  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (U.is_alias(out) && is_same_type<out_eT, typename T1::elem_type>::yes)
    {
    op_resize::apply_mat_inplace(out, new_n_rows, new_n_cols);
    }
  else
    {
    op_resize::apply_mat_noalias(out, E.M, new_n_rows, new_n_cols);
    }
  }



template<typename out_eT, typename T1>
inline
void
op_resize::apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to>, op_resize>& in)
  {
  coot_extra_debug_sigprint();

  const uword new_n_rows = in.aux_uword_a;
  const uword new_n_cols = in.aux_uword_b;

  const unwrap<T1> U(in.m.q);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  // Aliases aren't possible if the types are different (which is the only way an mtOp will get made).
  op_resize::apply_mat_noalias(out, E.M, new_n_rows, new_n_cols);
  }



template<typename eT>
inline
void
op_resize::apply_mat_inplace(Mat<eT>& A, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  if ((A.n_rows == new_n_rows) && (A.n_cols == new_n_cols))
    {
    return; // We're already the right size.
    }

  if (A.is_empty())
    {
    // There's no old matrix to copy, so just set everything to zeros.
    A.zeros(new_n_rows, new_n_cols);
    return;
    }

  Mat<eT> B;
  op_resize::apply_mat_noalias(B, A, new_n_rows, new_n_cols);
  A.steal_mem(B);
  }



template<typename out_eT, typename in_eT>
inline
void
op_resize::apply_mat_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  out.set_size(new_n_rows, new_n_cols);

  if ((new_n_rows > A.n_rows) || (new_n_cols > A.n_cols))
    {
    out.zeros();
    }

  if ((out.n_elem > 0) && (A.n_elem > 0))
    {
    const uword end_row = (std::min)(new_n_rows, A.n_rows) - 1;
    const uword end_col = (std::min)(new_n_cols, A.n_cols) - 1;

    out.submat(0, 0, end_row, end_col) = conv_to<Mat<out_eT>>::from(A.submat(0, 0, end_row, end_col));
    }
  }



template<typename T1>
inline
uword
op_resize::compute_n_rows(const Op<T1, op_resize>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return op.aux_uword_a;
  }



template<typename T1>
inline
uword
op_resize::compute_n_cols(const Op<T1, op_resize>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return op.aux_uword_b;
  }
