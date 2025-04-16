// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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
op_sum::apply(Mat<out_eT>& out, const Op<T1, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );

  // We have to consider type conversion carefully here.  If out_eT != T1::elem_type, then
  // the original operation was mtOp<out_eT, Op<T1, op_sum>, mtop_conv_to>, and so we want to
  // perform the conversion *after* computing the sum.
  //
  // On the other hand, T1 may be a conversion, giving the operation
  // Op<mtOp<T1::elem_type, T1, mtop_conv_to>, op_sum>.  In this situation, we want to perform
  // the conversion *before* computing the sum.  We can detect this condition if no_conv_unwrap
  // holds a different type than out_eT.

  // We can't perform two conversions though, so we'll greedily select the 'post' conversion if
  // it is happening.

  if (is_same_type<out_eT, typename T1::elem_type>::no)
    {
    // This is a post-sum conversion, so unwrap fully.
    const unwrap<T1> U(in.m);

    op_sum::apply_noalias(out, U.M, dim, true);
    }
  else
    {
    // This is a pre-sum conversion (or no conversion at all), so use a no-conv unwrap, which will
    // avoid performing a type conversion.
    const no_conv_unwrap<T1> U(in.m);

    // However, since there may be no conversion, we now have to consider aliases too.
    alias_wrapper<Mat<out_eT>, typename no_conv_unwrap<T1>::stored_type> W(out, U.M);
    op_sum::apply_noalias(W.use, U.M, dim, false);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_sum::apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if(dim == 0)
    {
    out.set_size(1, A.n_cols);
    }
  else
  if(dim == 1)
    {
    out.set_size(A.n_rows, 1);
    }

  if(A.n_elem == 0)
    {
    out.zeros();
    return;
    }


  coot_rt_t::sum(out.get_dev_mem(false), A.get_dev_mem(false),
                 A.n_rows, A.n_cols,
                 dim, post_conv_apply,
                 0, 1,
                 0, 0, A.n_rows);
  }



template<typename out_eT, typename in_eT>
inline
void
op_sum::apply_noalias(Mat<out_eT>& out, const subview<in_eT>& sv, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if(dim == 0)
    {
    out.set_size(1, sv.n_cols);
    }
  else
  if(dim == 1)
    {
    out.set_size(sv.n_rows, 1);
    }

  if(sv.n_elem == 0)
    {
    out.zeros();
    return;
    }

  coot_rt_t::sum(out.get_dev_mem(false), sv.m.get_dev_mem(false),
                 sv.n_rows, sv.n_cols,
                 dim, post_conv_apply,
                 0, 1,
                 sv.aux_row1, sv.aux_col1, sv.m.n_rows);
  }



template<typename T1>
inline
uword
op_sum::compute_n_rows(const Op<T1, op_sum>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);
  return (op.aux_uword_a == 0) ? 1 : in_n_rows;
  }



template<typename T1>
inline
uword
op_sum::compute_n_cols(const Op<T1, op_sum>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  return (op.aux_uword_a == 0) ? in_n_cols : 1;
  }
