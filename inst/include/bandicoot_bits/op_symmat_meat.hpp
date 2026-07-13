// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023-2026 Ryan Curtin (https://www.ratml.org)
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
op_symmatu::apply(Mat<out_eT>& out, const Op<T1, op_symmatu>& in)
  {
  coot_debug_sigprint();

  Proxy<Op<T1, op_symmatu>> P_in(in);
  coot_conform_check( (P_in.get_n_rows() != P_in.get_n_cols()), "symmatu(): given matrix must be square sized" );

  // We have to check for aliasing; if the T1 modifies the input matrix, we can't do the operation in-place.
  // NOTE: there is potential for optimization here; if the T1 *doesn't* modify the elements of the input, then we could do it in-place.
  alias_wrapper<Mat<out_eT>, Proxy<Op<T1, op_symmatu>>> A(out, P_in);
  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());

  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename T1>
inline
uword
op_symmatu::compute_n_rows(const Op<T1, op_symmatu>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_symmatu::compute_n_cols(const Op<T1, op_symmatu>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename out_eT, typename T1>
inline
void
op_symmatl::apply(Mat<out_eT>& out, const Op<T1, op_symmatl>& in)
  {
  coot_debug_sigprint();

  Proxy<Op<T1, op_symmatl>> P_in(in);
  coot_conform_check( (P_in.get_n_rows() != P_in.get_n_cols()), "symmatl(): given matrix must be square sized" );

  // We have to check for aliasing; if the T1 modifies the input matrix, we can't do the operation in-place.
  // NOTE: there is potential for optimization here; if the T1 *doesn't* modify the elements of the input, then we could do it in-place.
  alias_wrapper<Mat<out_eT>, Proxy<Op<T1, op_symmatl>>> A(out, P_in);
  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());

  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename T1>
inline
uword
op_symmatl::compute_n_rows(const Op<T1, op_symmatl>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_symmatl::compute_n_cols(const Op<T1, op_symmatl>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
