// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
mtop_any::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_any>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "any(): parameter 'dim' must be 0 or 1" );

  unwrap<T1> U(in.q);

  // Shortcut if the input is empty.
  if (U.M.n_elem == 0)
    {
    if (dim == 0)
      {
      out.set_size(1, 0);
      }
    else
      {
      out.set_size(0, 1);
      }

    return;
    }

  typedef typename T1::elem_type eT;
  apply_direct<eT, eT>(out, U.M, dim);
  }



// special handling of a conversion linked with an any()
template<typename T1, typename eT2>
inline
void
mtop_any::apply(Mat<uword>& out, const mtOp<uword, mtOp<eT2, T1, mtop_conv_to>, mtop_any>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "any(): parameter 'dim' must be 0 or 1" );

  unwrap<T1> U(in.q.q);

  // Shortcut if the input is empty.
  if (U.M.n_elem == 0)
    {
    if (dim == 0)
      {
      out.set_size(1, 0);
      }
    else
      {
      out.set_size(0, 1);
      }

    return;
    }

  typedef typename T1::elem_type eT;
  apply_direct<eT, eT2>(out, U.M, dim);
  }



template<typename T1, typename mtop_type>
inline
void
mtop_any::apply(Mat<uword>& out,
                const mtOp<uword, mtOp<uword, T1, mtop_type>, mtop_any>& in,
                const typename enable_if<is_same_type<mtop_type, mtop_conv_to>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "any(): parameter 'dim' must be 0 or 1" );

  // any( X != 0 ) --> any( X )
  const bool opt1 = (is_same_type<mtop_type, mtop_rel_noteq>::yes) && (in.q.aux == eT(0));
  // any( X >  0 ) --> any( X ) if eT is an unsigned integral type
  const bool opt2 = (is_same_type<mtop_type, mtop_rel_gt_post>::yes) && (is_signed<eT>::value == false) && (in.q.aux == eT(0));
  // any( 0 <  X ) --> any( X ) if eT is an unsigned integral type
  const bool opt3 = (is_same_type<mtop_type, mtop_rel_lt_pre>::yes) && (is_signed<eT>::value == false) && (in.q.aux == eT(0));

  // any( X <  0 ) --> zeros if eT is an unsigned integral type
  const bool opt4 = (is_same_type<mtop_type, mtop_rel_lt_post>::yes) && (is_signed<eT>::value == false) && (in.q.aux == eT(0));
  // any( 0 >  X ) --> zeros if eT is an unsigned integral type
  const bool opt5 = (is_same_type<mtop_type, mtop_rel_gt_pre>::yes) && (is_signed<eT>::value == false) && (in.q.aux == eT(0));

  if (opt1 || opt2 || opt3)
    {
    // Just call any() directly on the inner object.
    unwrap<T1> U(in.q.q);

    // Shortcut if the input is empty.
    if (U.M.n_elem == 0)
      {
      if (dim == 0)
        {
        out.set_size(1, 0);
        }
      else
        {
        out.set_size(0, 1);
        }

      return;
      }

    apply_direct<eT, eT>(out, U.M, dim);
    return;
    }
  else if (opt4 || opt5)
    {
    SizeProxy<T1> S(in.q.q);

    if (dim == 0)
      {
      out.zeros(1, S.get_n_cols());
      }
    else
      {
      out.zeros(S.get_n_rows(), 1);
      }

    return;
    }

  // No optimization available.
  unwrap<mtOp<uword, T1, mtop_type>> U(in.q);

  // Shortcut if the input is empty.
  if (U.M.n_elem == 0)
    {
    if (dim == 0)
      {
      out.set_size(1, 0);
      }
    else
      {
      out.set_size(0, 1);
      }

    return;
    }

  apply_direct<uword, uword>(out, U.M, dim);
  }



template<typename eT, typename eT2>
inline
void
mtop_any::apply_direct(Mat<uword>& out, const Mat<eT>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (((void*) &out) == (void*) &in)
    {
    // For aliases, we have to output into a temporary matrix.
    Mat<uword> tmp;
    apply_direct<eT, eT2>(tmp, in, dim);
    out.steal_mem(tmp);
    return;
    }

  if (dim == 0)
    {
    out.set_size(1, in.n_cols);
    coot_rt_t::any(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, eT2(0), twoway_kernel_id::rel_any_neq_colwise, true);
    }
  else
    {
    out.set_size(in.n_rows, 1);
    coot_rt_t::any(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, eT2(0), twoway_kernel_id::rel_any_neq_rowwise, false);
    }
  }



template<typename eT, typename eT2>
inline
void
mtop_any::apply_direct(Mat<uword>& out, const subview<eT>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted beforehand, and then we use the regular Mat implementation.
  Mat<eT> tmp(in);
  apply_direct<eT, eT2>(out, tmp, dim);
  }



template<typename T1>
inline
bool
mtop_any::any_vec(T1& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X);
  if (U.M.n_elem == 0)
    {
    return false;
    }

  return coot_rt_t::any_vec(U.M.get_dev_mem(false), U.M.n_elem, eT(0), twoway_kernel_id::rel_any_neq, twoway_kernel_id::rel_any_neq_small);
  }



template<typename out_eT, typename T1>
inline
bool
mtop_any::any_vec(const mtOp<out_eT, T1, mtop_any>& X)
  {
  coot_extra_debug_sigprint();

  // Apply to inner operation.
  return any_vec(X.q);
  }



template<typename eT2, typename T1>
inline
bool
mtop_any::any_vec(const mtOp<eT2, T1, mtop_conv_to>& op)
  {
  coot_extra_debug_sigprint();

  unwrap<T1> U(op.q);
  if (U.M.n_elem == 0)
    {
    return false;
    }

  return coot_rt_t::any_vec(U.M.get_dev_mem(false), U.M.n_elem, eT2(0), twoway_kernel_id::rel_any_neq, twoway_kernel_id::rel_any_neq_small);
  }



template<typename T1, typename mtop_type>
inline
bool
mtop_any::any_vec(const mtOp<uword, T1, mtop_type>& in,
                  const typename enable_if<is_same_type<mtop_type, mtop_any>::no>::result* junk1,
                  const typename enable_if<is_same_type<mtop_type, mtop_conv_to>::no>::result* junk2)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk1);
  coot_ignore(junk2);

  typedef typename T1::elem_type eT;

  // any( X != 0 ) --> any( X )
  const bool opt1 = (is_same_type<mtop_type, mtop_rel_noteq>::yes) && (in.aux == eT(0));
  // any( X >  0 ) --> any( X ) if eT is an unsigned integral type
  const bool opt2 = (is_same_type<mtop_type, mtop_rel_gt_post>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));
  // any( 0 <  X ) --> any( X ) if eT is an unsigned integral type
  const bool opt3 = (is_same_type<mtop_type, mtop_rel_lt_pre>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));

  // any( X == 0 ) --> !all( X )
  const bool opt4 = (is_same_type<mtop_type, mtop_rel_eq>::yes) && (in.aux == eT(0));
  // any( X <= 0 ) --> !all( X ) if eT is an unsigned integral type
  const bool opt5 = (is_same_type<mtop_type, mtop_rel_lteq_post>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));
  // any( 0 >= X ) --> !all( X ) if eT is an unsigned integral type
  const bool opt6 = (is_same_type<mtop_type, mtop_rel_gteq_pre>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));

  // any( X < 0 ) --> false if eT is an unsigned integral type
  const bool opt7 = (is_same_type<mtop_type, mtop_rel_lt_post>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));
  // any( 0 > X ) --> false if eT is an unsigned integral type
  const bool opt8 = (is_same_type<mtop_type, mtop_rel_gt_pre>::yes) && (is_signed<eT>::value == false) && (in.aux == eT(0));

  if (opt1 || opt2 || opt3)
    {
    return any_vec(in.q);
    }
  else if (opt4 || opt5 || opt6)
    {
    return !mtop_all::all_vec(in.q);
    }
  else if (opt7 || opt8)
    {
    return false;
    }

  // No optimization possible.
  unwrap<mtOp<uword, T1, mtop_type>> U(in);
  if (U.M.n_elem == 0)
    {
    return false;
    }

  return coot_rt_t::any_vec(U.M.get_dev_mem(false), U.M.n_elem, eT(0), twoway_kernel_id::rel_any_neq, twoway_kernel_id::rel_any_neq_small);
  }



template<typename out_eT, typename T1>
inline
uword
mtop_any::compute_n_rows(const mtOp<out_eT, T1, mtop_any>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (op.aux_uword_a == 0) ? 1 : in_n_rows;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_any::compute_n_cols(const mtOp<out_eT, T1, mtop_any>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (op.aux_uword_a == 0) ? in_n_cols : 1;
  }
