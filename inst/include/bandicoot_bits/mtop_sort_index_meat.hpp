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
mtop_sort_index::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_sort_index>& in)
  {
  coot_extra_debug_sigprint();

  const uword sort_type      = in.aux_uword_a;
  const uword is_stable_sort = in.aux_uword_b;

  const char* func_name = (is_stable_sort == 0) ? "sort_index()" : "stable_sort_index()";

  coot_debug_check( (sort_type > 1), std::string(func_name) + ": parameter 'sort_type' must be 0 or 1" );

  typedef typename T1::elem_type eT;

  // Note that T1 cannot be a Mat, Row, or Col---the next overload of apply() handles that.
  // Specifically that means that a new Mat is going to be created during the unwrap<> process.
  const unwrap<T1> U(in.q);
  // The kernels we have don't operate on subviews.
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  out.set_size(E.M.n_elem);

  // If there's nothing to do, don't do anything.
  if (E.M.n_elem == 0)
    {
    return;
    }

  coot_rt_t::sort_index_vec(out.get_dev_mem(false), const_cast<Mat<eT>&>(E.M).get_dev_mem(false), out.n_elem, sort_type, is_stable_sort);
  }



template<typename eT>
inline
void
mtop_sort_index::apply(Mat<uword>& out, const mtOp<uword, Mat<eT>, mtop_sort_index>& in)
  {
  coot_extra_debug_sigprint();

  const uword sort_type      = in.aux_uword_a;
  const uword is_stable_sort = in.aux_uword_b;

  const char* func_name = (is_stable_sort == 0) ? "sort_index()" : "stable_sort_index()";

  coot_debug_check( (sort_type > 1), std::string(func_name) + ": parameter 'sort_type' must be 0 or 1" );

  // No unwrapping is necessary, but we do need to copy the input to a temporary matrix, since sort_index() will destroy the input matrix.
  Mat<eT> tmp(in.q);

  out.set_size(tmp.n_elem);

  // If there's nothing to do, don't do anything.
  if (tmp.n_elem == 0)
    {
    return;
    }

  coot_rt_t::sort_index_vec(out.get_dev_mem(false), tmp.get_dev_mem(false), out.n_elem, sort_type, is_stable_sort);
  }



template<typename out_eT, typename T1>
inline
uword
mtop_sort_index::compute_n_rows(const mtOp<out_eT, T1, mtop_sort_index>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  return in_n_rows * in_n_cols;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_sort_index::compute_n_cols(const mtOp<out_eT, T1, mtop_sort_index>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return 1;
  }
