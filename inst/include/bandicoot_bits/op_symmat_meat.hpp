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
op_symmat::apply(Mat<out_eT>& out, const Op<T1, op_symmat>& in)
  {
  coot_extra_debug_sigprint();

  const uword lower = in.aux_uword_a;

  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (lower)
    {
    coot_debug_check( (E.M.n_rows != E.M.n_cols), "symmatl(): given matrix must be square sized" );
    }
  else
    {
    coot_debug_check( (E.M.n_rows != E.M.n_cols), "symmatu(): given matrix must be square sized" );
    }

  // It's okay if `out` is an alias of `E.M`; the kernel can be run in-place with no problems.
  out.set_size(E.M.n_rows, E.M.n_cols);

  if (E.M.n_elem == 0)
    {
    // Nothing to do---quit early.
    return;
    }

  coot_rt_t::symmat(out.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_rows, lower);
  }



template<typename out_eT, typename T1>
inline
void
op_symmat::apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to>, op_symmat>& in)
  {
  coot_extra_debug_sigprint();

  const uword lower = in.aux_uword_a;

  const unwrap<T1> U(in.m.q);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (lower)
    {
    coot_debug_check( (E.M.n_rows != E.M.n_cols), "symmatl(): given matrix must be square sized" );
    }
  else
    {
    coot_debug_check( (E.M.n_rows != E.M.n_cols), "symmatu(): given matrix must be square sized" );
    }

  // Aliases are not possible if a conversion is involved.
  out.set_size(E.M.n_rows, E.M.n_cols);

  if (E.M.n_elem == 0)
    {
    // Nothing to do---quit early.
    return;
    }

  coot_rt_t::symmat(out.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_rows, lower);
  }



template<typename T1>
inline
uword
op_symmat::compute_n_rows(const Op<T1, op_symmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_symmat::compute_n_cols(const Op<T1, op_symmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
