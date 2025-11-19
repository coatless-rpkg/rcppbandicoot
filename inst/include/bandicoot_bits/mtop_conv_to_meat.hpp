// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (http://www.ratml.org
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
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // Unwrap the inner operation fully.
  const unwrap<T1> U(X.q);

  out.set_size(U.M.n_rows, U.M.n_cols);

  coot_rt_t::copy_mat(out.get_dev_mem(false), U.get_dev_mem(false),
                      out.n_rows, out.n_cols,
                      0, 0, out.n_rows,
                      U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // Unwrap the inner operation fully.
  const unwrap_cube<T1> U(X.q);

  out.set_size(U.M.n_rows, U.M.n_cols, U.M.n_slices);

  coot_rt_t::copy_cube(out.get_dev_mem(false), U.get_dev_mem(false),
                       out.n_rows, out.n_cols, out.n_slices,
                       0, 0, 0, out.n_rows, out.n_cols,
                       U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



// TODO: this overload might only be applicable for ops that support conversions during the op
// TODO: decide whether all ops should support conversions too internally
template<typename out_eT, typename T1, typename op_type>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, Op<T1, op_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  op_type::apply(out, X.q);
  }



template<typename out_eT, typename T1, typename eop_type>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, eOp<T1, eop_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  out.set_size(X.q.m.get_n_rows(), X.q.m.get_n_cols());

  // Apply the operation specifically into the different output type.
  eop_type::apply(out, X.q);
  }



template<typename out_eT, typename T1, typename T2, typename eglue_type>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, eGlue<T1, T2, eglue_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  out.set_size(X.q.get_n_rows(), X.q.get_n_cols());

  // Apply the operation specifically into the different output type.
  eglue_type::apply(out, X.q);
  }



template<typename out_eT, typename T1, typename eop_type>
inline
void
mtop_conv_to::apply(Cube<out_eT>& out, const mtOpCube<out_eT, eOpCube<T1, eop_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  out.set_size(X.q.get_n_rows(), X.q.get_n_cols(), X.q.get_n_slices());

  // Apply the operation specifically into the different output type.
  eop_type::apply(out, X.q);
  }



template<typename out_eT, typename T1, typename T2, typename eglue_type>
inline
void
mtop_conv_to::apply(Cube<out_eT>& out, const mtOpCube<out_eT, eGlueCube<T1, T2, eglue_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  out.set_size(X.q.get_n_rows(), X.q.get_n_cols(), X.q.get_n_slices());

  // Apply the operation specifically into the different output type.
  eglue_type::apply(out, X.q);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_plus(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.q);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_plus_array,
                     out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     0, 0, out.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_minus(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.q);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_minus_array,
                     out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     0, 0, out.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_times(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // We have to actually perform the conversion here.
  Mat<out_eT> converted(X.q);
  Mat<out_eT> tmp(out);
  tmp *= converted;

  out.steal_mem(tmp);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_schur(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.q);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                     out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     0, 0, out.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_div(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.q);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_div_array,
                     out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     0, 0, out.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_plus(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(X.q);

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_minus(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(X.q);

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_schur(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(X.q);

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_div(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(X.q);

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), U.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename out_eT, typename T1>
inline
uword
mtop_conv_to::compute_n_rows(const mtOp<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(X);
  coot_ignore(in_n_cols);

  // mtop_conv_to does not change the size of the input.
  return in_n_rows;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_conv_to::compute_n_cols(const mtOp<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(X);
  coot_ignore(in_n_rows);

  // mtop_conv_to does not change the size of the input.
  return in_n_cols;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_conv_to::compute_n_rows(const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(X);
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);

  // mtop_conv_to does not change the size of the input.
  return in_n_rows;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_conv_to::compute_n_cols(const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(X);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_slices);

  // mtop_conv_to does not change the size of the input.
  return in_n_cols;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_conv_to::compute_n_slices(const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(X);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  // mtop_conv_to does not change the size of the input.
  return in_n_slices;
  }
