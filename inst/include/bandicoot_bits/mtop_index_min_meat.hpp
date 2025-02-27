// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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
mtop_index_min::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_index_min>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "index_min(): parameter 'dim' must be 0 or 1" );

  const unwrap<T1> U(in.q);

  if (U.is_alias(out) == false)
    {
    mtop_index_min::apply_noalias(out, U.M, dim);
    }
  else
    {
    Mat<uword> tmp;
    mtop_index_min::apply_noalias(tmp, U.M, dim);
    out.steal_mem(tmp);
    }
  }



template<typename eT>
inline
void
mtop_index_min::apply_noalias(Mat<uword>& out, const Mat<eT>& A, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(1, A.n_cols);
    }
  else
    {
    out.set_size(A.n_rows, 1);
    }

  if (A.n_elem == 0)
    {
    out.zeros();
    return;
    }

  coot_rt_t::index_min(out.get_dev_mem(false), A.get_dev_mem(false),
                       A.n_rows, A.n_cols, dim,
                       0, 1,
                       0, 0, A.n_rows);
  }



template<typename eT>
inline
void
mtop_index_min::apply_noalias(Mat<uword>& out, const subview<eT>& sv, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(1, sv.n_cols);
    }
  else if (dim == 1)
    {
    out.set_size(sv.n_rows, 1);
    }

  if (sv.n_elem == 0)
    {
    out.zeros();
    return;
    }

  coot_rt_t::index_min(out.get_dev_mem(false), sv.m.get_dev_mem(false),
                       sv.n_rows, sv.n_cols, dim,
                       0, 1,
                       sv.aux_row1, sv.aux_col1, sv.m.n_rows);
  }



template<typename T1>
inline
uword
mtop_index_min::compute_n_rows(const mtOp<uword, T1, mtop_index_min>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);
  return (op.aux_uword_a == 0) ? 1 : in_n_rows;
  }



template<typename T1>
inline
uword
mtop_index_min::compute_n_cols(const mtOp<uword, T1, mtop_index_min>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  return (op.aux_uword_a == 0) ? in_n_cols : 1;
  }



template<typename T1>
inline
uword
mtop_index_min::apply_direct(const T1& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in);
  const Mat<typename T1::elem_type>& A = U.M;

  return coot_rt_t::index_min_vec(A.get_dev_mem(false), A.n_elem);
  }
