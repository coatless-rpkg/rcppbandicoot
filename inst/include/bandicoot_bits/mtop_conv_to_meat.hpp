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
  coot_debug_sigprint();

  const Proxy<mtOp<out_eT, T1, mtop_conv_to>> P_in(X);
  const inexact_alias_wrapper<Mat<out_eT>, Proxy<mtOp<out_eT, T1, mtop_conv_to>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X)
  {
  coot_debug_sigprint();

  const Proxy<mtOpCube<out_eT, T1, mtop_conv_to>> P_in(X);
  const inexact_alias_wrapper<Cube<out_eT>, Proxy<mtOpCube<out_eT, T1, mtop_conv_to>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols(), P_in.get_n_slices());
  coot_rt_t::copy(make_proxy(A.use), P_in);
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
