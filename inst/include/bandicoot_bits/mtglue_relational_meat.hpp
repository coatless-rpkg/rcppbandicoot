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



template<typename T1, typename T2>
inline
void
mtglue_rel_lt::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_lt>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator<");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_lt_array, "operator<");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_lt::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_lt>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_lt::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_lt>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_gt::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_gt>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator>");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_gt_array, "operator>");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_gt::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_gt>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_gt::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_gt>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_lteq::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_lteq>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator<=");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_lteq_array, "operator<=");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_lteq::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_lteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_lteq::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_lteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_gteq::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_gteq>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator>=");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_gteq_array, "operator>=");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_gteq::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_gteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_gteq::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_gteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_eq::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_eq>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator==");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_eq_array, "operator==");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_eq::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_eq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_eq::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_eq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_noteq::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_noteq>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator!=");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_neq_array, "operator!=");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_noteq::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_noteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_noteq::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_noteq>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_and::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_and>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator&&");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_and_array, "operator&&");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_and::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_and>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_and::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_and>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename T1, typename T2>
inline
void
mtglue_rel_or::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_or>& X)
  {
  coot_extra_debug_sigprint();

  // Check that sizes are equal without doing any work.
  SizeProxy<T1> SPA(X.A);
  SizeProxy<T2> SPB(X.B);

  coot_assert_same_size(SPA.get_n_rows(), SPA.get_n_cols(), SPB.get_n_rows(), SPB.get_n_cols(), "operator||");

  out.set_size(SPA.get_n_rows(), SPA.get_n_cols());

  // Now unwrap the two arguments to prepare for the backend call.
  typedef typename T1::elem_type eT1;
  unwrap<T1> UA(X.A);
  extract_subview<typename unwrap<T1>::stored_type> EA(UA.M);
  copy_alias<eT1> CA(EA.M, out);

  typedef typename T2::elem_type eT2;
  unwrap<T2> UB(X.B);
  extract_subview<typename unwrap<T2>::stored_type> EB(UB.M);
  copy_alias<eT2> CB(EB.M, out);

  coot_rt_t::relational_array_op(out.get_dev_mem(false), CA.M.get_dev_mem(false), CB.M.get_dev_mem(false), CA.M.n_elem, twoway_kernel_id::rel_or_array, "operator||");
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_or::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_or>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
mtglue_rel_or::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_or>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }
