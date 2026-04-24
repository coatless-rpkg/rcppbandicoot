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



template<typename mtglue_type>
template<typename T1, typename T2>
inline
void
mtglue_rel_core<mtglue_type>::apply(Mat<uword>& out, const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>& X)
  {
  coot_debug_sigprint();

  // TODO: size check for T1/T2

  const Proxy<mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>> P_in(X);
  const inexact_alias_wrapper<Mat<uword>, Proxy<mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
void
mtglue_rel_core<mtglue_type>::apply(Cube<uword>& out, const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& X)
  {
  coot_debug_sigprint();

  // TODO: size check for T1/T2

  const Proxy<mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>> P_in(X);
  const inexact_alias_wrapper<Cube<uword>, Proxy<mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols(), P_in.get_n_slices());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
uword
mtglue_rel_core<mtglue_type>::compute_n_rows(const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
uword
mtglue_rel_core<mtglue_type>::compute_n_cols(const mtGlue<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
uword
mtglue_rel_core<mtglue_type>::compute_n_rows(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
uword
mtglue_rel_core<mtglue_type>::compute_n_cols(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename mtglue_type>
template<typename T1, typename T2>
inline
uword
mtglue_rel_core<mtglue_type>::compute_n_slices(const mtGlueCube<uword, T1, T2, mtglue_rel_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }
