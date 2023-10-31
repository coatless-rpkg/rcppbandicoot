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
op_median::apply(Mat<out_eT>& out, const Op<T1, op_median>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  // Note that T1 cannot be a Mat, Row, or Col---the next overload of apply() handles that.
  // Specifically that means that a new Mat is going to be created during the unwrap<> process.
  unwrap<T1> U(in.m);
  // The kernels we have don't operate on subviews.
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  const uword dim = in.aux_uword_a;
  // We can drop the `const` from E.M because we know that the held matrix is a temporary and thus we can reuse it.
  apply_direct(out, const_cast<Mat<eT>&>(E.M), dim);
  }



template<typename out_eT, typename eT>
inline
void
op_median::apply(Mat<out_eT>& out, const Op<Mat<eT>, op_median>& in)
  {
  coot_extra_debug_sigprint();

  // No unwrapping is necessary, but we do need to copy the input to a temporary matrix, since median() will destroy (sort) the input matrix.
  Mat<eT> tmp(in.m);
  const uword dim = in.aux_uword_a;
  apply_direct(out, tmp, dim);
  }



template<typename out_eT, typename in_eT>
inline
void
op_median::apply_direct(Mat<out_eT>& out, Mat<in_eT>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

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

  coot_rt_t::median(out.get_dev_mem(false), in.get_dev_mem(false),
                    in.n_rows, in.n_cols,
                    dim,
                    0, 1,
                    0, 0, in.n_rows);
  }



template<typename T1>
inline
typename T1::elem_type
op_median::median_all(const T1& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X); // This will cause the creation of a new matrix, which we will use as a temporary.
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (E.M.n_elem == 0)
    {
    return eT(0);
    }

  return coot_rt_t::median_vec(E.M.get_dev_mem(false), E.M.n_elem);
  }



template<typename eT>
inline
eT
op_median::median_all(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  if (X.n_elem == 0)
    {
    return eT(0);
    }

  // We need to copy the matrix to a temporary, so that we can sort and compute the median.
  Mat<eT> tmp(X);
  return coot_rt_t::median_vec(tmp.get_dev_mem(false), tmp.n_elem);
  }



template<typename T1>
inline
uword
op_median::compute_n_rows(const Op<T1, op_median>& op, const uword in_n_rows, const uword in_n_cols)
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
op_median::compute_n_cols(const Op<T1, op_median>& op, const uword in_n_rows, const uword in_n_cols)
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
