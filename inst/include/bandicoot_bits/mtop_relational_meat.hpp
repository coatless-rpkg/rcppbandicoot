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



template<typename mtop_type>
template<typename T1>
inline
void
mtop_rel_core<mtop_type>::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_rel_core<mtop_type> >& X)
  {
  coot_debug_sigprint();

  const Proxy<mtOp<uword, T1, mtop_rel_core<mtop_type> >> P_in(X);
  const inexact_alias_wrapper<Mat<uword>, Proxy<mtOp<uword, T1, mtop_rel_core<mtop_type> >>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename mtop_type>
template<typename T1>
inline
void
mtop_rel_core<mtop_type>::apply(Cube<uword>& out, const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& X)
  {
  coot_debug_sigprint();

  const Proxy<mtOpCube<uword, T1, mtop_rel_core<mtop_type> >> P_in(X);
  const inexact_alias_wrapper<Cube<uword>, Proxy<mtOpCube<uword, T1, mtop_rel_core<mtop_type> >>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols(), P_in.get_n_slices());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename mtop_type>
template<typename T1>
inline
uword
mtop_rel_core<mtop_type>::compute_n_rows(const mtOp<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename mtop_type>
template<typename T1>
inline
uword
mtop_rel_core<mtop_type>::compute_n_cols(const mtOp<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename mtop_type>
template<typename T1>
inline
uword
mtop_rel_core<mtop_type>::compute_n_rows(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);

  return in_n_rows;
  }



template<typename mtop_type>
template<typename T1>
inline
uword
mtop_rel_core<mtop_type>::compute_n_cols(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_slices);

  return in_n_cols;
  }



template<typename mtop_type>
template<typename T1>
inline
uword
mtop_rel_core<mtop_type>::compute_n_slices(const mtOpCube<uword, T1, mtop_rel_core<mtop_type> >& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return in_n_slices;
  }
