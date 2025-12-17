// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (https://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//
// mtglue_mixed_times
//



// matrix multiplication with different element types
template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_times::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type in_eT1;
  typedef typename T2::elem_type in_eT2;

  // For mixed matrix multiplication, we have to convert both results to the output type.
  const partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>> tmp1(mtOp<out_eT, T1, mtop_conv_to>(X.A));
  const partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>> tmp2(mtOp<out_eT, T2, mtop_conv_to>(X.B));

  typedef typename partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::stored_type PT1;
  typedef typename partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::stored_type PT2;

  const PT1& A = tmp1.M;
  const PT2& B = tmp2.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val()) : out_eT(0);

  alias_wrapper<Mat<out_eT>, PT1, PT2> W(out, A, B);
  glue_times::apply
    <
    out_eT,
    PT1,
    PT2,
    partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::do_trans,
    partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::do_trans,
    (partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::do_times || partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::do_times)
    >
    (W.use, A, B, alpha);
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_times::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_times::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



//
// mtglue_mixed_plus
//



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_plus::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& X)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> UA(X.A.get_ref());
  unwrap<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Mat<out_eT>, typename unwrap<T1>::stored_type, typename unwrap<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_plus::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_plus_array,
                     A.get_dev_mem(false),
                     UA.get_dev_mem(false),
                     UB.get_dev_mem(false),
                     A.use.n_rows, A.use.n_cols,
                     0, 0, A.use.n_rows,
                     UA.get_row_offset(), UA.get_col_offset(), UA.get_M_n_rows(),
                     UB.get_row_offset(), UB.get_col_offset(), UB.get_M_n_rows());
  }



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_plus::apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& X)
  {
  coot_extra_debug_sigprint();

  unwrap_cube<T1> UA(X.A.get_ref());
  unwrap_cube<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Cube<out_eT>, typename unwrap_cube<T1>::stored_type, typename unwrap_cube<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_plus::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols, UA.M.n_slices);

  coot_rt_t::eop_cube(threeway_kernel_id::equ_array_plus_array_cube,
                      A.get_dev_mem(false),
                      UA.get_dev_mem(false),
                      UB.get_dev_mem(false),
                      A.use.n_rows, A.use.n_cols, A.use.n_slices,
                      0, 0, 0, A.use.n_rows, A.use.n_cols,
                      UA.get_row_offset(), UA.get_col_offset(), UA.get_slice_offset(), UA.get_M_n_rows(), UA.get_M_n_cols(),
                      UB.get_row_offset(), UB.get_col_offset(), UB.get_slice_offset(), UB.get_M_n_rows(), UB.get_M_n_cols());
  }




template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_plus::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_plus::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_plus::compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_plus::compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_plus::compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_plus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }



//
// mtglue_mixed_minus
//



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_minus::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& X)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> UA(X.A.get_ref());
  unwrap<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Mat<out_eT>, typename unwrap<T1>::stored_type, typename unwrap<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_minus::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_minus_array,
                     A.get_dev_mem(false),
                     UA.get_dev_mem(false),
                     UB.get_dev_mem(false),
                     A.use.n_rows, A.use.n_cols,
                     0, 0, A.use.n_rows,
                     UA.get_row_offset(), UA.get_col_offset(), UA.get_M_n_rows(),
                     UB.get_row_offset(), UB.get_col_offset(), UB.get_M_n_rows());
  }



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_minus::apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& X)
  {
  coot_extra_debug_sigprint();

  unwrap_cube<T1> UA(X.A.get_ref());
  unwrap_cube<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Cube<out_eT>, typename unwrap_cube<T1>::stored_type, typename unwrap_cube<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_minus::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols, UA.M.n_slices);

  coot_rt_t::eop_cube(threeway_kernel_id::equ_array_minus_array_cube,
                      A.get_dev_mem(false),
                      UA.get_dev_mem(false),
                      UB.get_dev_mem(false),
                      A.use.n_rows, A.use.n_cols, A.use.n_slices,
                      0, 0, 0, A.use.n_rows, A.use.n_cols,
                      UA.get_row_offset(), UA.get_col_offset(), UA.get_slice_offset(), UA.get_M_n_rows(), UA.get_M_n_cols(),
                      UB.get_row_offset(), UB.get_col_offset(), UB.get_slice_offset(), UB.get_M_n_rows(), UB.get_M_n_cols());
  }




template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_minus::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_minus::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_minus::compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_minus::compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_minus::compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_minus>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }



//
// mtglue_mixed_div
//



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_div::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& X)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> UA(X.A.get_ref());
  unwrap<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Mat<out_eT>, typename unwrap<T1>::stored_type, typename unwrap<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_div::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_div_array,
                     A.get_dev_mem(false),
                     UA.get_dev_mem(false),
                     UB.get_dev_mem(false),
                     A.use.n_rows, A.use.n_cols,
                     0, 0, A.use.n_rows,
                     UA.get_row_offset(), UA.get_col_offset(), UA.get_M_n_rows(),
                     UB.get_row_offset(), UB.get_col_offset(), UB.get_M_n_rows());
  }



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_div::apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& X)
  {
  coot_extra_debug_sigprint();

  unwrap_cube<T1> UA(X.A.get_ref());
  unwrap_cube<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Cube<out_eT>, typename unwrap_cube<T1>::stored_type, typename unwrap_cube<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_div::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols, UA.M.n_slices);

  coot_rt_t::eop_cube(threeway_kernel_id::equ_array_div_array_cube,
                      A.get_dev_mem(false),
                      UA.get_dev_mem(false),
                      UB.get_dev_mem(false),
                      A.use.n_rows, A.use.n_cols, A.use.n_slices,
                      0, 0, 0, A.use.n_rows, A.use.n_cols,
                      UA.get_row_offset(), UA.get_col_offset(), UA.get_slice_offset(), UA.get_M_n_rows(), UA.get_M_n_cols(),
                      UB.get_row_offset(), UB.get_col_offset(), UB.get_slice_offset(), UB.get_M_n_rows(), UB.get_M_n_cols());
  }




template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_div::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_div::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_div::compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_div::compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_div::compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_div>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }



//
// mtglue_mixed_schur
//



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_schur::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& X)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> UA(X.A.get_ref());
  unwrap<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Mat<out_eT>, typename unwrap<T1>::stored_type, typename unwrap<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_schur::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols);

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                     A.get_dev_mem(false),
                     UA.get_dev_mem(false),
                     UB.get_dev_mem(false),
                     A.use.n_rows, A.use.n_cols,
                     0, 0, A.use.n_rows,
                     UA.get_row_offset(), UA.get_col_offset(), UA.get_M_n_rows(),
                     UB.get_row_offset(), UB.get_col_offset(), UB.get_M_n_rows());
  }



template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_schur::apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& X)
  {
  coot_extra_debug_sigprint();

  unwrap_cube<T1> UA(X.A.get_ref());
  unwrap_cube<T2> UB(X.B.get_ref());

  // Make sure we are not operating on an alias.
  alias_wrapper<Cube<out_eT>, typename unwrap_cube<T1>::stored_type, typename unwrap_cube<T2>::stored_type> A(out, UA.M, UB.M);

  coot_assert_same_size( UA.M, UB.M, "mtglue_mixed_schur::apply()" );

  A.use.set_size(UA.M.n_rows, UA.M.n_cols, UA.M.n_slices);

  coot_rt_t::eop_cube(threeway_kernel_id::equ_array_mul_array_cube,
                      A.get_dev_mem(false),
                      UA.get_dev_mem(false),
                      UB.get_dev_mem(false),
                      A.use.n_rows, A.use.n_cols, A.use.n_slices,
                      0, 0, 0, A.use.n_rows, A.use.n_cols,
                      UA.get_row_offset(), UA.get_col_offset(), UA.get_slice_offset(), UA.get_M_n_rows(), UA.get_M_n_cols(),
                      UB.get_row_offset(), UB.get_col_offset(), UB.get_slice_offset(), UB.get_M_n_rows(), UB.get_M_n_cols());
  }




template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_schur::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_schur::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_schur::compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_schur::compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_schur::compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_schur>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }
