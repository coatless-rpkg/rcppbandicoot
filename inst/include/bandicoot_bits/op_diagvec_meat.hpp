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
op_diagvec::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_diagvec>& in)
  {
  coot_extra_debug_sigprint();

  // Extract diagonal id.
  const sword k = (in.aux_uword_b == 0) ? sword(in.aux_uword_a) : -sword(in.aux_uword_a);

  unwrap<T1> U(in.m);
  op_diagvec::apply_direct(out, U.M, k);
  }



template<typename out_eT, typename T1>
inline
void
op_diagvec::apply(Mat<out_eT>& out, const Op<T1, op_diagvec>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // Extract diagonal id.
  const sword k = (in.aux_uword_b == 0) ? sword(in.aux_uword_a) : -sword(in.aux_uword_a);

  // If the types are not the same, we have to force a conversion.
  mtOp<out_eT, T1, mtop_conv_to> mtop(in.m);
  unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
  op_diagvec::apply_direct(out, U.M, k);
  }



template<typename eT>
inline
void
op_diagvec::apply_direct(Mat<eT>& out, const Mat<eT>& in, const sword k)
  {
  coot_extra_debug_sigprint();

  // If out and in are the same matrix, we can't do the operation in-place.
  if (&out == &in)
    {
    Mat<eT> tmp(in);
    op_diagvec::apply_direct(out, tmp, k);
    return;
    }

  const uword len = (std::min)(in.n_rows, in.n_cols) - std::abs(k);

  out.set_size(len, 1);

  const uword row_offset = (k < 0) ? uword(-k) : 0;
  const uword col_offset = (k > 0) ? uword( k) : 0;

  const uword mem_offset = row_offset + col_offset * in.n_rows;

  coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                      1, len,
                      0, 0, 1,
                      mem_offset, 0, in.n_rows + 1);
  }



template<typename eT>
inline
void
op_diagvec::apply_direct(Mat<eT>& out, const subview<eT>& in, const sword k)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted.
  Mat<eT> tmp(in);
  op_diagvec::apply_direct(out, tmp, k);
  }



template<typename T1>
inline
uword
op_diagvec::compute_n_rows(const Op<T1, op_diagvec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (std::min)(in_n_rows, in_n_cols) - op.aux_uword_a;
  }



template<typename T1>
inline
uword
op_diagvec::compute_n_cols(const Op<T1, op_diagvec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return 1;
  }
