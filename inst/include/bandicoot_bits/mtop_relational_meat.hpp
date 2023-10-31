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



template<typename T1>
inline
void
mtop_rel_lt_pre::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lt_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar < X
  // this is equivalent to X > scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_gt_scalar, "operator<");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_lt_pre::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lt_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar < eT2(X)
  // this is equivalent to X > scalar, with scalar as eT2

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_gt_scalar, "operator<");
  }



template<typename T1>
inline
uword
mtop_rel_lt_pre::compute_n_rows(const mtOp<uword, T1, mtop_rel_lt_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_lt_pre::compute_n_cols(const mtOp<uword, T1, mtop_rel_lt_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_lt_post::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lt_post>& X)
  {
  coot_extra_debug_sigprint();

  // X < scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_lt_scalar, "operator<");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_lt_post::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lt_post>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) < scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_lt_scalar, "operator<");
  }



template<typename T1>
inline
uword
mtop_rel_lt_post::compute_n_rows(const mtOp<uword, T1, mtop_rel_lt_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_lt_post::compute_n_cols(const mtOp<uword, T1, mtop_rel_lt_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_gt_pre::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gt_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar > X
  // this is equivalent to X < scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_lt_scalar, "operator>");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_gt_pre::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gt_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar > eT2(X)
  // this is equivalent to X < scalar, with scalar as eT2

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_lt_scalar, "operator>");
  }



template<typename T1>
inline
uword
mtop_rel_gt_pre::compute_n_rows(const mtOp<uword, T1, mtop_rel_gt_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_gt_pre::compute_n_cols(const mtOp<uword, T1, mtop_rel_gt_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_gt_post::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gt_post>& X)
  {
  coot_extra_debug_sigprint();

  // X > scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_gt_scalar, "operator>");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_gt_post::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gt_post>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) > scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_gt_scalar, "operator>");
  }



template<typename T1>
inline
uword
mtop_rel_gt_post::compute_n_rows(const mtOp<uword, T1, mtop_rel_gt_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_gt_post::compute_n_cols(const mtOp<uword, T1, mtop_rel_gt_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_lteq_pre::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lteq_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar <= X
  // rewritten to X >= scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_gteq_scalar, "operator<=");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_lteq_pre::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_lteq_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar <= eT2(X)
  // this is equivalent to X >= scalar, with scalar as eT2

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_gteq_scalar, "operator<=");
  }



template<typename T1>
inline
uword
mtop_rel_lteq_pre::compute_n_rows(const mtOp<uword, T1, mtop_rel_lteq_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_lteq_pre::compute_n_cols(const mtOp<uword, T1, mtop_rel_lteq_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_lteq_post::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_lteq_post>& X)
  {
  coot_extra_debug_sigprint();

  // X <= scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_lteq_scalar, "operator<=");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_lteq_post::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1,
mtop_conv_to>, mtop_rel_lteq_post>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) <= scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_lteq_scalar, "operator<=");
  }



template<typename T1>
inline
uword
mtop_rel_lteq_post::compute_n_rows(const mtOp<uword, T1, mtop_rel_lteq_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_lteq_post::compute_n_cols(const mtOp<uword, T1, mtop_rel_lteq_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }




template<typename T1>
inline
void
mtop_rel_gteq_pre::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gteq_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar >= X
  // rewritten to X <= scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_lteq_scalar, "operator>=");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_gteq_pre::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gteq_pre>& X)
  {
  coot_extra_debug_sigprint();

  // scalar >= eT2(X)
  // this is equivalent to X <= scalar, with scalar as eT2

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_lteq_scalar, "operator>=");
  }



template<typename T1>
inline
uword
mtop_rel_gteq_pre::compute_n_rows(const mtOp<uword, T1, mtop_rel_gteq_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_gteq_pre::compute_n_cols(const mtOp<uword, T1, mtop_rel_gteq_pre>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_gteq_post::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_gteq_post>& X)
  {
  coot_extra_debug_sigprint();

  // X >= scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_gteq_scalar, "operator>=");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_gteq_post::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_gteq_post>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) > scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_gteq_scalar, "operator>=");
  }



template<typename T1>
inline
uword
mtop_rel_gteq_post::compute_n_rows(const mtOp<uword, T1, mtop_rel_gteq_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_gteq_post::compute_n_cols(const mtOp<uword, T1, mtop_rel_gteq_post>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_eq::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_eq>& X)
  {
  coot_extra_debug_sigprint();

  // X == scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_eq_scalar, "operator==");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_eq::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_eq>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) == scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_eq_scalar, "operator==");
  }



template<typename T1>
inline
uword
mtop_rel_eq::compute_n_rows(const mtOp<uword, T1, mtop_rel_eq>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_eq::compute_n_cols(const mtOp<uword, T1, mtop_rel_eq>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
void
mtop_rel_noteq::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_noteq>& X)
  {
  coot_extra_debug_sigprint();

  // X != scalar

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT(X.aux), twoway_kernel_id::rel_neq_scalar, "operator!=");
  }



template<typename eT2, typename T1>
inline
void
mtop_rel_noteq::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_rel_noteq>& X)
  {
  coot_extra_debug_sigprint();

  // eT2(X) != scalar

  typedef typename T1::elem_type eT1;
  unwrap<T1> U(X.q.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  copy_alias<eT1> C(E.M, out);

  out.set_size(C.M.n_rows, C.M.n_cols);
  coot_rt_t::relational_scalar_op(out.get_dev_mem(false), C.M.get_dev_mem(false), C.M.n_elem, eT2(X.aux), twoway_kernel_id::rel_neq_scalar, "operator!=");
  }



template<typename T1>
inline
uword
mtop_rel_noteq::compute_n_rows(const mtOp<uword, T1, mtop_rel_noteq>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
mtop_rel_noteq::compute_n_cols(const mtOp<uword, T1, mtop_rel_noteq>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
