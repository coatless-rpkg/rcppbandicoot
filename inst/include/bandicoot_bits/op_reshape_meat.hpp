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



template<typename out_eT, typename T1>
inline
void
op_reshape::apply(Mat<out_eT>& out, const Op<T1, op_reshape>& in)
  {
  coot_extra_debug_sigprint();

  // If we are reshaping to an empty matrix, just clear the output and return.
  if (in.aux_uword_a == 0 || in.aux_uword_b == 0)
    {
    out.set_size(in.aux_uword_a, in.aux_uword_b);
    return;
    }

  unwrap<T1> U(in.m);
  op_reshape::apply_direct(out, U.M, in.aux_uword_a, in.aux_uword_b);
  }



template<typename out_eT, typename T1>
inline
void
op_reshape::apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to>, op_reshape>& in)
  {
  coot_extra_debug_sigprint();

  Op<T1, op_reshape> op_tmp(in.m.q, in.aux_uword_a, in.aux_uword_b);
  op_reshape::apply(out, op_tmp);
  }



template<typename eT>
inline
void
op_reshape::apply_direct(Mat<eT>& out, const Mat<eT>& in, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword new_n_elem = new_n_rows * new_n_cols;

  if ((&out) == (&in))
    {
    // If the number of elements is the same and this is an alias, this is the
    // easiest case.  We can leave the memory as-is.
    if ((new_n_rows * new_n_cols) == in.n_elem)
      {
      access::rw(out.n_rows) = new_n_rows;
      access::rw(out.n_cols) = new_n_cols;
      }
    else
      {
      Mat<eT> tmp(new_n_rows, new_n_cols);
      op_reshape::apply_direct(tmp, in, new_n_rows, new_n_cols);
      out.steal_mem(tmp);
      }
    }
  else
    {
    out.set_size(new_n_rows, new_n_cols);

    if (new_n_elem > in.n_elem)
      {
      // Set all the memory to zeros, since some zero elements will be needed.
      coot_rt_t::fill(out.get_dev_mem(false), eT(0), new_n_rows, new_n_cols, 0, 0, new_n_rows);
      }

    if (in.n_elem > 0)
      {
      // We treat both out and in as column vectors here.
      const uword elems_to_copy = (std::min)(new_n_elem, in.n_elem);
      coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                          elems_to_copy, 1,
                          0, 0, elems_to_copy,
                          0, 0, elems_to_copy);
      }
    }
  }



template<typename out_eT, typename eT>
inline
void
op_reshape::apply_direct(Mat<out_eT>& out, const Mat<eT>& in, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword new_n_elem = new_n_rows * new_n_cols;

  // We can't have an alias because the type is changing.
  out.set_size(new_n_rows, new_n_cols);
  if (new_n_elem > in.n_elem)
    {
    // Set all the memory to zeros, since some zero elements will be needed.
    coot_rt_t::fill(out.get_dev_mem(false), out_eT(0), new_n_rows, new_n_cols, 0, 0, new_n_rows);
    }

  if (in.n_elem > 0)
    {
    // We treat both out and in as column vectors here.
    const uword elems_to_copy = (std::min)(new_n_elem, in.n_elem);
    coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                        elems_to_copy, 1,
                        0, 0, elems_to_copy,
                        0, 0, elems_to_copy);
    }
  }



template<typename eT>
inline
void
op_reshape::apply_direct(Mat<eT>& out, const subview<eT>& in, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  if ((&out) == (&in.m))
    {
    // We don't have support for safe in-place subview-to-matrix extractions, so
    // extract the subview.
    Mat<eT> tmp(in);
    op_reshape::apply_direct(out, tmp, new_n_rows, new_n_cols);
    return;
    }

  const uword new_n_elem = new_n_rows * new_n_cols;

  out.set_size(new_n_rows, new_n_cols);

  if (new_n_elem > in.n_elem)
    {
    // Set all the memory to zeros, since some zero elements will be needed.
    coot_rt_t::fill(out.get_dev_mem(false), eT(0),
                    new_n_elem - in.n_elem, 1,
                    in.n_elem, 0, new_n_elem - in.n_elem);

    if (in.n_elem > 0)
      {
      coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                          in.n_rows, in.n_cols,
                          0, 0, in.n_rows /* intentionally not out.n_rows */,
                          in.aux_row1, in.aux_col1, in.m.n_rows);
      }
    }
  else if (new_n_elem == in.n_elem && in.n_elem > 0)
    {
    coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                        in.n_rows, in.n_cols,
                        0, 0, in.n_rows /* intentionally not out.n_rows */,
                        in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    // We don't have a way to only copy some elements out of a subview, so
    // extract the subview entirely and then reshape it...
    Mat<eT> tmp(in);
    op_reshape::apply_direct(out, tmp, new_n_rows, new_n_cols);
    return;
    }
  }



template<typename out_eT, typename eT>
inline
void
op_reshape::apply_direct(Mat<out_eT>& out, const subview<eT>& in, const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword new_n_elem = new_n_rows * new_n_cols;

  out.set_size(new_n_rows, new_n_cols);

  if (new_n_elem > in.n_elem)
    {
    // Set all the memory to zeros, since some zero elements will be needed.
    coot_rt_t::fill(out.get_dev_mem(false), out_eT(0),
                    new_n_elem - in.n_elem, 1,
                    in.n_elem, 0, new_n_rows - in.n_elem);
    if (in.n_elem > 0)
      {
      coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                          in.n_rows, in.n_cols,
                          0, 0, in.n_rows /* intentionally not out.n_rows */,
                          in.aux_row1, in.aux_col1, in.m.n_rows);
      }
    }
  else if (new_n_elem == in.n_elem && in.n_elem > 0)
    {
    coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                        in.n_rows, in.n_cols,
                        0, 0, in.n_rows /* intentionally not out.n_rows */,
                        in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    // We don't have a way to only copy some elements out of a subview, so
    // extract the subview entirely and then reshape it...
    Mat<eT> tmp(in);
    op_reshape::apply_direct(out, tmp, new_n_rows, new_n_cols);
    return;
    }
  }



template<typename T1>
inline
uword
op_reshape::compute_n_rows(const Op<T1, op_reshape>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return op.aux_uword_a;
  }



template<typename T1>
inline
uword
op_reshape::compute_n_cols(const Op<T1, op_reshape>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return op.aux_uword_b;
  }
