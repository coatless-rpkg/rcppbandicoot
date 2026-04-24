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



template<typename out_eT, typename T1>
inline
void
op_diagmat::apply(Mat<out_eT>& out, const Op<T1, op_diagmat>& in)
  {
  coot_debug_sigprint();

  const Proxy<T1> P(in.m);
  const alias_wrapper<Mat<out_eT>, Proxy<T1>> A(out, P);
  if (P.get_n_rows() == 1 || P.get_n_cols() == 1)
    {
    A.use.zeros(P.get_n_elem(), P.get_n_elem());
    diagview<out_eT> d = A.use.diag();
    coot_rt_t::copy(make_proxy(d), P);
    }
  else
    {
    // We can only have a diagview of an existing matrix, so unwrap the proxy into a temporary matrix.
    Mat<out_eT> tmp(P.get_n_rows(), P.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P);
    A.use.zeros(tmp.n_rows, tmp.n_cols);

    diagview<out_eT> d_in = tmp.diag();
    diagview<out_eT> d = A.use.diag();

    coot_rt_t::copy(make_proxy(d), make_proxy(d_in));
    }
  }



template<typename T1>
inline
uword
op_diagmat::compute_n_rows(const Op<T1, op_diagmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1 || in_n_cols == 1)
    return (std::max)(in_n_rows, in_n_cols);
  else
    return in_n_rows;
  }



template<typename T1>
inline
uword
op_diagmat::compute_n_cols(const Op<T1, op_diagmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1 || in_n_cols == 1)
    return (std::max)(in_n_rows, in_n_cols);
  else
    return in_n_cols;
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat2::apply(Mat<out_eT>& out, const Op<T1, op_diagmat2>& in)
  {
  coot_debug_sigprint();

  const sword k = (in.aux_uword_b % 2 == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));
  const bool swap = (in.aux_uword_b >= 2);

  const Proxy<T1> P(in.m);
  const alias_wrapper<Mat<out_eT>, Proxy<T1>> A(out, P);
  if (P.get_n_rows() == 1 || P.get_n_cols() == 1)
    {
    A.use.zeros(P.get_n_elem() + std::abs(k), P.get_n_elem() + std::abs(k));
    diagview<out_eT> d = A.use.diag(k);
    coot_rt_t::copy(make_proxy(d), P);
    }
  else
    {
    // We can only have a diagview of an existing matrix, so unwrap the proxy into a temporary matrix.
    Mat<out_eT> tmp(P.get_n_rows(), P.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P);
    A.use.zeros(tmp.n_rows, tmp.n_cols);

    diagview<out_eT> d_in = swap ? tmp.diag(-k) : tmp.diag(k);
    diagview<out_eT> d = A.use.diag(k);

    coot_rt_t::copy(make_proxy(d), make_proxy(d_in));
    }
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat2::apply(Mat<out_eT>& out, const Op<Op<T1, op_htrans2>, op_diagmat2>& in)
  {
  coot_debug_sigprint();

  const sword k = (in.aux_uword_b % 2 == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));
  const bool swap = (in.aux_uword_b >= 2);

  const eOp<T1, eop_scalar_times> E(in.m.m, in.m.aux_a);
  const Proxy<eOp<T1, eop_scalar_times>> P(in.m.m);
  const alias_wrapper<Mat<out_eT>, Proxy<eOp<T1, eop_scalar_times>>> A(out, P);
  if (P.get_n_rows() == 1 || P.get_n_cols() == 1)
    {
    A.use.zeros(P.get_n_elem() + std::abs(k), P.get_n_elem() + std::abs(k));
    diagview<out_eT> d = A.use.diag(k);
    coot_rt_t::copy(make_proxy(d), P);
    }
  else
    {
    // We can only have a diagview of an existing matrix, so unwrap the proxy into a temporary matrix.
    Mat<out_eT> tmp(P.get_n_rows(), P.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P);
    A.use.zeros(tmp.n_rows, tmp.n_cols);

    diagview<out_eT> d_in = swap ? tmp.diag(-k) : tmp.diag(k);
    diagview<out_eT> d = A.use.diag(k);

    coot_rt_t::copy(make_proxy(d), make_proxy(d_in));
    }
  }



template<typename T1>
inline
uword
op_diagmat2::compute_n_rows(const Op<T1, op_diagmat2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (in_n_rows == 1 || in_n_cols == 1)
    return (std::max)(in_n_rows, in_n_cols) + op.aux_uword_a;
  else
    return in_n_rows;
  }



template<typename T1>
inline
uword
op_diagmat2::compute_n_cols(const Op<T1, op_diagmat2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (in_n_rows == 1 || in_n_cols == 1)
    return (std::max)(in_n_rows, in_n_cols) + op.aux_uword_a;
  else
    return in_n_cols;
  }
