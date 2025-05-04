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
mtop_index_max::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_index_max>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "index_max(): parameter 'dim' must be 0 or 1" );

  const unwrap<T1> U(in.q);

  alias_wrapper<Mat<uword>, typename unwrap<T1>::stored_type> W(out, in.q);
  mtop_index_max::apply_noalias(W.use, U.M, dim);
  }



template<typename eT>
inline
void
mtop_index_max::apply_noalias(Mat<uword>& out, const Mat<eT>& A, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(A.n_rows == 0 ? 0 : 1, A.n_cols);
    }
  else
    {
    out.set_size(A.n_rows, A.n_cols == 0 ? 0 : 1);
    }

  if (A.n_elem == 0)
    {
    out.zeros();
    return;
    }

  coot_rt_t::index_max(out.get_dev_mem(false), A.get_dev_mem(false),
                       A.n_rows, A.n_cols, dim,
                       0, 1,
                       0, 0, A.n_rows);
  }



template<typename eT>
inline
void
mtop_index_max::apply_noalias(Mat<uword>& out, const subview<eT>& sv, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(sv.n_rows == 0 ? 0 : 1, sv.n_cols);
    }
  else if (dim == 1)
    {
    out.set_size(sv.n_rows, sv.n_cols == 0 ? 0 : 1);
    }

  if (sv.n_elem == 0)
    {
    out.zeros();
    return;
    }

  coot_rt_t::index_max(out.get_dev_mem(false), sv.m.get_dev_mem(false),
                       sv.n_rows, sv.n_cols, dim,
                       0, 1,
                       sv.aux_row1, sv.aux_col1, sv.m.n_rows);
  }



template<typename T1>
inline
uword
mtop_index_max::compute_n_rows(const mtOp<uword, T1, mtop_index_max>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);
  return (op.aux_uword_a == 0 && in_n_rows > 0) ? 1 : in_n_rows;
  }



template<typename T1>
inline
uword
mtop_index_max::compute_n_cols(const mtOp<uword, T1, mtop_index_max>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  return (op.aux_uword_a == 0 && in_n_cols > 0) ? in_n_cols : 1;
  }



template<typename T1>
inline
void
mtop_index_max::apply(Cube<uword>& out, const mtOpCube<uword, T1, mtop_index_max>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 2), "index_max(): parameter 'dim' must be 0, 1, or 2" );

  const unwrap_cube<T1> U(in.q);
  const extract_subcube<typename unwrap_cube<T1>::stored_type> E(U.M);

  alias_wrapper<Cube<uword>, Cube<typename unwrap_cube<T1>::stored_type::elem_type>> W(out, in.q);
  mtop_index_max::apply_noalias(W.use, E.M, dim);
  }



template<typename eT>
inline
void
mtop_index_max::apply_noalias(Cube<uword>& out, const Cube<eT>& A, const uword dim)
  {
  coot_extra_debug_sigprint();

  out.set_size((dim == 0 && A.n_rows > 0) ? 1 : A.n_rows,
               (dim == 1 && A.n_cols > 0) ? 1 : A.n_cols,
               (dim == 2 && A.n_slices > 0) ? 1 : A.n_slices);

  if (A.n_elem == 0)
    {
    out.zeros();
    return;
    }

  // When we are computing the minimum along a dimension for a Cube (not a subcube!),
  // we can make two simple optimizations that allow us to reuse the matrix kernels.
  if (dim == 0)
    {
    // If we are computing the minimum value in each row, we can treat the cube as
    // a matrix of size n_rows x (n_cols * n_slices).

    coot_rt_t::index_max(out.get_dev_mem(false), A.get_dev_mem(false),
                         A.n_rows, (A.n_cols * A.n_slices), dim,
                         0, 1,
                         0, 0, A.n_rows);
    }
  else if (dim == 2)
    {
    // If we are computing the minimum value in each slice, we can treat the cube as
    // a matrix of size (n_rows * n_cols) x n_slices.

    coot_rt_t::index_max(out.get_dev_mem(false), A.get_dev_mem(false),
                         (A.n_rows * A.n_cols), A.n_slices, dim,
                         0, 1,
                         0, 0, A.n_rows * A.n_cols);
    }
  else
    {
    // If we are computing the minimum value in each column, the situation is slightly
    // more complicated---we need a kernel specific to cubes.

    coot_rt_t::index_max_cube_col(out.get_dev_mem(false), A.get_dev_mem(false),
                                  A.n_rows, A.n_cols, A.n_slices);
    }
  }



template<typename T1>
inline
uword
mtop_index_max::compute_n_rows(const mtOpCube<uword, T1, mtop_index_max>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);
  return (op.aux_uword_a == 0 && in_n_rows > 0) ? 1 : in_n_rows;
  }



template<typename T1>
inline
uword
mtop_index_max::compute_n_cols(const mtOpCube<uword, T1, mtop_index_max>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_slices);
  return (op.aux_uword_a == 1 && in_n_cols > 0) ? 1 : in_n_cols;
  }



template<typename T1>
inline
uword
mtop_index_max::compute_n_slices(const mtOpCube<uword, T1, mtop_index_max>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return (op.aux_uword_a == 2 && in_n_slices > 0) ? 1 : in_n_slices;
  }



template<typename T1>
inline
uword
mtop_index_max::apply_direct(const Base<typename T1::elem_type, T1>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.get_ref());
  const Mat<typename T1::elem_type>& A = U.M;

  return coot_rt_t::index_max_vec(A.get_dev_mem(false), A.n_elem);
  }



template<typename T1>
inline
uword
mtop_index_max::apply_direct(const BaseCube<typename T1::elem_type, T1>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(in.get_ref());
  const Cube<typename T1::elem_type>& A = U.M;

  return coot_rt_t::index_max_vec(A.get_dev_mem(false), A.n_elem);
  }
