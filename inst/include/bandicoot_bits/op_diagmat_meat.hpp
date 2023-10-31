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
op_diagmat::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_diagmat>& in)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> U(in.m);
  op_diagmat::apply_direct(out, U.M);
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat::apply(Mat<out_eT>& out, const Op<T1, op_diagmat>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // If the types are not the same, we have to force a conversion.
  mtOp<out_eT, T1, mtop_conv_to> mtop(in.m);
  unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
  op_diagmat::apply_direct(out, U.M);
  }



template<typename eT>
inline
void
op_diagmat::apply_direct(Mat<eT>& out, const Mat<eT>& in)
  {
  coot_extra_debug_sigprint();

  if (in.n_rows == 1 || in.n_cols == 1)
    {
    out.zeros(in.n_elem, in.n_elem);
    out.diag() = in;
    }
  else
    {
    out.zeros(in.n_rows, in.n_cols);
    out.diag() = in.diag();
    }
  }



template<typename eT>
inline
void
op_diagmat::apply_direct(Mat<eT>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted.
  Mat<eT> tmp(in);
  if (tmp.n_rows == 1 || tmp.n_cols == 1)
    {
    out.zeros(tmp.n_elem, tmp.n_elem);
    out.diag() = tmp;
    }
  else
    {
    out.zeros(tmp.n_rows, tmp.n_cols);
    out.diag() = tmp.diag();
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



template<typename T1>
inline
void
op_diagmat2::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_diagmat2>& in)
  {
  coot_extra_debug_sigprint();

  const sword k = (in.aux_uword_b % 2 == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));
  const bool swap = (in.aux_uword_b >= 2);

  unwrap<T1> U(in.m);
  op_diagmat2::apply_direct(out, U.M, k, swap);
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat2::apply(Mat<out_eT>& out, const Op<T1, op_diagmat2>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  const sword k = (in.aux_uword_b % 2 == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));
  const bool swap = (in.aux_uword_b >= 2);

  // If the types are not the same, we have to force a conversion.
  mtOp<out_eT, T1, mtop_conv_to> mtop(in.m);
  unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
  op_diagmat2::apply_direct(out, U.M, k, swap);
  }



template<typename T1>
inline
void
op_diagmat2::apply(Mat<typename T1::elem_type>& out, const Op<Op<T1, op_htrans2>, op_diagmat2>& in)
  {
  coot_extra_debug_sigprint();

  const sword k = (in.aux_uword_b == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));

  unwrap<T1> U(in.m.m);
  op_diagmat2::apply_direct(out, U.M, k, true /* implicit transpose */);
  // Now multiply by the scalar.
  out *= in.m.aux;
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat2::apply(Mat<out_eT>& out, const Op<Op<T1, op_htrans2>, op_diagmat2>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  const sword k = (in.aux_uword_b == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));

  // If the types are not the same, we have to force a conversion.
  mtOp<out_eT, T1, mtop_conv_to> mtop(in.m.m);
  unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
  op_diagmat2::apply_direct(out, U.M, k, true /* implicit transpose */);
  // Now multiply by the scalar.
  out *= in.m.aux;
  }



template<typename eT>
inline
void
op_diagmat2::apply_direct(Mat<eT>& out, const Mat<eT>& in, const sword k, const bool swap)
  {
  coot_extra_debug_sigprint();

  if (in.n_rows == 1 || in.n_cols == 1)
    {
    out.zeros(in.n_elem + std::abs(k), in.n_elem + std::abs(k));
    out.diag(k) = in;
    }
  else
    {
    out.zeros(in.n_rows, in.n_cols);
    if (swap)
      {
      out.diag(k) = in.diag(-k);
      }
    else
      {
      out.diag(k) = in.diag(k);
      }
    }
  }



template<typename eT>
inline
void
op_diagmat2::apply_direct(Mat<eT>& out, const subview<eT>& in, const sword k, const bool swap)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted.
  Mat<eT> tmp(in);
  if (tmp.n_rows == 1 || tmp.n_cols == 1)
    {
    out.zeros(tmp.n_elem + std::abs(k), tmp.n_elem + std::abs(k));
    out.diag(k) = tmp;
    }
  else
    {
    out.zeros(tmp.n_rows, tmp.n_cols);
    if (swap)
      {
      out.diag(k) = in.diag(-k);
      }
    else
      {
      out.diag(k) = in.diag(k);
      }
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
