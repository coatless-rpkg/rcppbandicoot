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
op_repmat::apply_noalias(Mat<out_eT>& out, const T1& X, const uword copies_per_row, const uword copies_per_col)
  {
  coot_extra_debug_sigprint();

  const uword new_n_rows = X.n_rows * copies_per_row;
  const uword new_n_cols = X.n_cols * copies_per_col;

  out.set_size(new_n_rows, new_n_cols);

  // If the matrix has zero elements, no need to do anything.
  if (new_n_rows == 0 || new_n_cols == 0)
    {
    return;
    }

  coot_rt_t::repmat(X.get_dev_mem(false), out.get_dev_mem(false), X.n_rows, X.n_cols, copies_per_row, copies_per_col);
  }



template<typename out_eT, typename T1>
inline
void
op_repmat::apply(Mat<out_eT>& out, const Op<T1, op_repmat>& in)
  {
  coot_extra_debug_sigprint();

  const uword copies_per_row = in.aux_uword_a;
  const uword copies_per_col = in.aux_uword_b;

  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (U.is_alias(out))
    {
    // Skip if there is nothing to do.
    if (copies_per_row == 1 && copies_per_col == 1 && std::is_same<out_eT, typename T1::elem_type>::value)
      {
      return;
      }

    Mat<typename T1::elem_type> tmp(E.M);
    apply_noalias(out, tmp, copies_per_row, copies_per_col);
    }
  else
    {
    apply_noalias(out, E.M, copies_per_row, copies_per_col);
    }
  }



template<typename out_eT, typename T1>
inline
void
op_repmat::apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to>, op_repmat>& in)
  {
  coot_extra_debug_sigprint();

  const uword copies_per_row = in.aux_uword_a;
  const uword copies_per_col = in.aux_uword_b;

  const unwrap<T1> U(in.m.q);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (U.is_alias(out))
    {
    // Skip if there is nothing to do.
    if (copies_per_row == 1 && copies_per_col == 1 && std::is_same<out_eT, typename T1::elem_type>::value)
      {
      return;
      }

    Mat<typename T1::elem_type> tmp(E.M);
    apply_noalias(out, tmp, copies_per_row, copies_per_col);
    }
  else
    {
    apply_noalias(out, E.M, copies_per_row, copies_per_col);
    }
  }



template<typename T1>
inline
uword
op_repmat::compute_n_rows(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);
  return op.aux_uword_a * in_n_rows;
  }



template<typename T1>
inline
uword
op_repmat::compute_n_cols(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  return op.aux_uword_b * in_n_cols;
  }
