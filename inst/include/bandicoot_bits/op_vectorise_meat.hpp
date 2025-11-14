// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2021-2022 Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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
op_vectorise_col::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_col>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);

  if (U.M.n_elem == 0)
    {
    out.set_size(0, 1);
    return;
    }

  op_vectorise_col::apply_direct(out, U.M);
  }



template<typename eT>
inline
void
op_vectorise_col::apply_direct(Mat<eT>& out, const Mat<eT>& in, const bool output_is_row)
  {
  coot_extra_debug_sigprint();

  if(is_alias(out, in))
    {
    // We can just reshape and call it a day.
    if (!output_is_row)
      {
      out.set_size(out.n_elem, 1);  // set_size() doesn't destroy data as long as the number of elements in the matrix remains the same
      }
    else
      {
      out.set_size(1, out.n_elem);
      }
    }
  else
    {
    if (!output_is_row)
      {
      out.set_size(in.n_elem, 1);
      }
    else
      {
      out.set_size(1, in.n_elem);
      }

    coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                        // logically, we treat `out` as the same size as the input
                        in.n_rows, in.n_cols,
                        0, 0, in.n_rows,
                        0, 0, in.n_rows);
    }
  }



template<typename eT>
inline
void
op_vectorise_col::apply_direct(Mat<eT>& out, const subview<eT>& in, const bool output_is_row)
  {
  coot_extra_debug_sigprint();

  if(is_alias(out, in))
    {
    // We have to extract the subview.
    Mat<eT> tmp(in);
    op_vectorise_col::apply_direct(out, tmp, output_is_row);
    }
  else
    {
    if (!output_is_row)
      {
      out.set_size(in.n_elem, 1);
      }
    else
      {
      out.set_size(1, in.n_elem);
      }

    coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                        // logically, we treat `out` as the same size as the input
                        in.n_rows, in.n_cols,
                        0, 0, in.n_rows,
                        in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_vectorise_col::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const bool output_is_row)
  {
  coot_extra_debug_sigprint();

  if (!output_is_row)
    {
    out.set_size(in.n_elem, 1);
    }
  else
    {
    out.set_size(1, in.n_elem);
    }

  coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                      // logically, we treat `out` as the same size as the input
                      in.n_rows, in.n_cols,
                      0, 0, in.n_rows,
                      0, 0, in.n_rows);
  }



template<typename out_eT, typename in_eT>
inline
void
op_vectorise_col::apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const bool output_is_row)
  {
  coot_extra_debug_sigprint();

  if (!output_is_row)
    {
    out.set_size(in.n_elem, 1);
    }
  else
    {
    out.set_size(1, in.n_elem);
    }

  coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                      // logically, we treat `out` as the same size as the input
                      in.n_rows, in.n_cols,
                      0, 0, in.n_rows,
                      in.aux_row1, in.aux_col1, in.m.n_rows);
  }



template<typename T1>
inline
uword
op_vectorise_col::compute_n_rows(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  return in_n_rows * in_n_cols;
  }



template<typename T1>
inline
uword
op_vectorise_col::compute_n_cols(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return 1;
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_all::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_all>& in)
  {
  coot_extra_debug_sigprint();

  SizeProxy<T1> S(in.m);

  if (in.aux_uword_a == 0)
    {
    if (S.get_n_elem() == 0)
      {
      out.set_size(1, 0);
      return;
      }

    unwrap<T1> U(in.m);
    op_vectorise_col::apply_direct(out, U.M);
    }
  else
    {
    // See if we can use op_vectorise_col anyway, which we can do if the object is already a vector.
    if (S.get_n_rows() == 1 || S.get_n_cols() == 1)
      {
      if (S.get_n_elem() == 0)
        {
        out.set_size(0, 1);
        return;
        }

      unwrap<T1> U(in.m);
      op_vectorise_col::apply_direct(out, U.M, true /* use in row vector mode */);
      }
    else
      {
      op_vectorise_row::apply_direct(out, in.m);
      }
    }
  }



template<typename T1>
inline
uword
op_vectorise_all::compute_n_rows(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (op.aux_uword_a == 0)
    return in_n_rows * in_n_cols;
  else
    return 1;
  }



template<typename T1>
inline
uword
op_vectorise_all::compute_n_cols(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (op.aux_uword_a == 0)
    return 1;
  else
    return in_n_rows * in_n_cols;
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_row::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_row>& in)
  {
  coot_extra_debug_sigprint();

  op_vectorise_row::apply_direct(out, in.m);
  }



template<typename T1>
inline
void
op_vectorise_row::apply_direct(Mat<typename T1::elem_type>& out, const T1& expr)
  {
  coot_extra_debug_sigprint();

  // Row-wise vectorisation is equivalent to a transpose followed by a vectorisation.

  // TODO: select htrans/strans based on complex elements or not
  // Using op_htrans as part of the unwrap may combine the htrans with some earlier operations in the expression.
  unwrap<Op<T1, op_htrans>> U(Op<T1, op_htrans>(expr, 0, 0));

  // If U.M is an object we created during unwrapping, steal the memory and set the size.
  // Otherwise, copy U.M.
  // TODO: this is not correct!
  if (is_Mat<T1>::value || is_subview<T1>::value)
    {
    // If `expr` is some type of matrix, then unwrap<T1> just stores the matrix itself.
    // That's not a temporary, and we can't steal its memory---we have to copy it.
    out.set_size(1, U.M.n_elem);
    coot_rt_t::copy_mat(out.get_dev_mem(false), U.get_dev_mem(false),
                        // logically, we treat `out` as the same size as the input
                        U.M.n_rows, U.M.n_cols,
                        0, 0, U.M.n_rows,
                        U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  else
    {
    // We must have created a temporary matrix to perform the operation, and so we can just steal its memory.
    const uword new_n_rows = U.M.n_elem;
    out.steal_mem(U.M);
    out.set_size(1, new_n_rows);
    }
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_row::apply_direct(Mat<out_eT>& out, const T1& expr, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // Row-wise vectorisation is equivalent to a transpose followed by a vectorisation.

  // TODO: select htrans/strans based on complex elements or not
  // Using op_htrans as part of the unwrap may combine the htrans with some earlier operations in the expression.
  unwrap<Op<T1, op_htrans>> U(Op<T1, op_htrans>(expr, 0, 0));

  // A conversion operation is always necessary when the type is different.
  out.set_size(1, U.M.n_elem);
  coot_rt_t::copy_mat(out.get_dev_mem(false), U.get_dev_mem(false),
                      // logically, we treat `out` as the same size as the input
                      U.M.n_rows, U.M.n_cols,
                      0, 0, U.M.n_rows,
                      U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename T1>
inline
uword
op_vectorise_row::compute_n_rows(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return 1;
  }



template<typename T1>
inline
uword
op_vectorise_row::compute_n_cols(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  return in_n_rows * in_n_cols;
  }



template<typename T1>
inline
void
op_vectorise_cube_col::apply(Mat<typename T1::elem_type>& out, const CubeToMatOp<T1, op_vectorise_cube_col>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(in.m);

  if(U.is_alias(out))
    {
    // output matrix is the same as the input matrix
    out.set_size(out.n_elem, 1);  // set_size() doesn't destroy data as long as the number of elements in the matrix remains the same
    }
  else
    {
    out.set_size(U.M.n_elem, 1);

    coot_rt_t::copy_cube(out.get_dev_mem(false), U.get_dev_mem(false),
                         // logically, we treat `out` as the same size as the input
                         U.M.n_rows, U.M.n_cols, U.M.n_slices,
                         0, 0, 0, U.M.n_rows, U.M.n_cols,
                         U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
    }
  }



template<typename T1>
inline
uword
op_vectorise_cube_col::compute_n_rows(const CubeToMatOp<T1, op_vectorise_cube_col>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);

  return in_n_rows * in_n_cols * in_n_slices;
  }



template<typename T1>
inline
uword
op_vectorise_cube_col::compute_n_cols(const CubeToMatOp<T1, op_vectorise_cube_col>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);

  return 1;
  }
