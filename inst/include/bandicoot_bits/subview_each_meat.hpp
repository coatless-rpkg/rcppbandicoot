// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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



template<typename parent, unsigned int mode>
inline
subview_each_common<parent, mode>::subview_each_common(const parent& in_P)
  : P(in_P)
  {
  coot_debug_sigprint();
  }



template<typename parent, unsigned int mode>
inline
void
subview_each_common<parent, mode>::check_size(const uword p_n_rows, const uword p_n_cols, const uword copies_per_row, const uword copies_per_col, const uword in_n_rows, const uword in_n_cols)
  {
  if(mode == 0)
    {
    if( (copies_per_row * in_n_rows != p_n_rows) || (in_n_cols != 1) )
      {
      coot_stop_logic_error( incompat_size_string(p_n_rows, p_n_cols, in_n_rows, in_n_cols) );
      }
    }
  else
    {
    if( (in_n_rows != 1) || (copies_per_col * in_n_cols != p_n_cols) )
      {
      coot_stop_logic_error( incompat_size_string(p_n_rows, p_n_cols, in_n_rows, in_n_cols) );
      }
    }
  }



template<typename parent, unsigned int mode>
inline
const std::string
subview_each_common<parent, mode>::incompat_size_string(const uword p_n_rows, const uword p_n_cols, const uword in_n_rows, const uword in_n_cols)
  {
  std::ostringstream tmp;

  if(mode == 0)
    {
    tmp << "each_col(): incompatible size; expected " << p_n_rows << "x1" << ", got " << in_n_rows << 'x' << in_n_cols;
    }
  else
    {
    tmp << "each_row(): incompatible size; expected 1x" << p_n_cols << ", got " << in_n_rows << 'x' << in_n_cols;
    }

  return tmp.str();
  }



//
// subview_each1
//



template<typename parent, unsigned int mode>
inline
subview_each1<parent, mode>::~subview_each1()
  {
  coot_debug_sigprint();
  }



template<typename parent, unsigned int mode>
inline
subview_each1<parent, mode>::subview_each1(const parent& in_P)
  : subview_each_common<parent, mode>::subview_each_common(in_P)
  {
  coot_debug_sigprint();
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  parent& P_ref = access::rw(subview_each_common<parent, mode>::P);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_ref.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? P_ref.n_cols : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T1, op_repmat>> P(op);

  subview_each_common<parent, mode>::check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P.get_n_rows(), P.P.get_n_cols());
  if (P.is_empty())
    {
    return;
    }

  alias_wrapper<parent, Proxy<Op<T1, op_repmat>>> A(P_ref, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    }
  else
    {
    coot_rt_t::copy(make_proxy(P_ref), P);
    }
  }




template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator+=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  parent& P_ref = access::rw(subview_each_common<parent, mode>::P);

  const uword copies_per_row = (mode == 1) ? P_ref.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? P_ref.n_cols : 1;

  const Op<T1, op_repmat> op_inner(in.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T1, op_repmat>, eglue_plus> glue(P_ref, op_inner);
  const Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_plus>> P(glue);

  subview_each_common<parent, mode>::check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return;
    }

  alias_wrapper<parent, Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_plus>>> A(P_ref, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    }
  else
    {
    coot_rt_t::copy(make_proxy(P_ref), P);
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator-=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  parent& P_ref = access::rw(subview_each_common<parent, mode>::P);

  const uword copies_per_row = (mode == 1) ? P_ref.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? P_ref.n_cols : 1;

  const Op<T1, op_repmat> op_inner(in.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T1, op_repmat>, eglue_minus> glue(P_ref, op_inner);
  const Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_minus>> P(glue);

  subview_each_common<parent, mode>::check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return;
    }

  alias_wrapper<parent, Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_minus>>> A(P_ref, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    }
  else
    {
    coot_rt_t::copy(make_proxy(P_ref), P);
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator%=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  parent& P_ref = access::rw(subview_each_common<parent, mode>::P);

  const uword copies_per_row = (mode == 1) ? P_ref.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? P_ref.n_cols : 1;

  const Op<T1, op_repmat> op_inner(in.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T1, op_repmat>, eglue_schur> glue(P_ref, op_inner);
  const Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_schur>> P(glue);

  subview_each_common<parent, mode>::check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return;
    }

  alias_wrapper<parent, Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_schur>>> A(P_ref, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    }
  else
    {
    coot_rt_t::copy(make_proxy(P_ref), P);
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator/=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  parent& P_ref = access::rw(subview_each_common<parent, mode>::P);

  const uword copies_per_row = (mode == 1) ? P_ref.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? P_ref.n_cols : 1;

  const Op<T1, op_repmat> op_inner(in.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T1, op_repmat>, eglue_div> glue(P_ref, op_inner);
  const Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_div>> P(glue);

  subview_each_common<parent, mode>::check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return;
    }

  alias_wrapper<parent, Proxy<eGlue<parent, Op<T1, op_repmat>, eglue_div>>> A(P_ref, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    }
  else
    {
    coot_rt_t::copy(make_proxy(P_ref), P);
    }
  }



//
// subview_each2
//



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent, mode, TB>::~subview_each2()
  {
  coot_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent, mode, TB>::subview_each2(const parent& in_P, const Base<uword, TB>& in_indices)
  : subview_each_common<parent, mode>::subview_each_common(in_P)
  , base_indices(in_indices)
  {
  coot_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
void
subview_each2<parent, mode, TB>::check_indices(const Mat<uword>& indices) const
  {
  if(mode == 0)
    {
    coot_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_col(): list of indices must be a vector" );
    }
  else
    {
    coot_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_row(): list of indices must be a vector" );
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(*this);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T1, op_repmat>> P_op(op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());
  if (P_op.is_empty())
    {
    return;
    }

  if (P_op.is_alias(subview_each_common<parent, mode>::P))
    {
    // alias_wrapper does not set the size correctly for a subview_elem2, so create the temporary matrix manually.
    Mat<typename parent::elem_type> tmp(P_op.get_n_rows(), P_op.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P_op);
    coot_rt_t::copy(P_lhs, make_proxy(tmp));
    }
  else
    {
    coot_rt_t::copy(P_lhs, P_op);
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator+=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(*this);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);

  // We will create the eGlue Proxy manually so as to not make P_lhs twice...
  // but in order to do that, we need to ensure that we have the right type for the Op proxy.
  typedef typename Proxy_glue_type< Op<T1, op_repmat>, subview_each2<parent, mode, TB> >::result Proxy_op_type;
  const Proxy< Proxy_op_type > P_op(op);
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T1, op_repmat>, eglue_plus>> P_glue(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());
  if (P_glue.is_empty())
    {
    return;
    }

  if (P_op.is_alias(subview_each_common<parent, mode>::P))
    {
    // alias_wrapper does not set the size correctly for a subview_elem2, so create the temporary matrix manually.
    Mat<typename parent::elem_type> tmp(P_op.get_n_rows(), P_op.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P_glue);
    coot_rt_t::copy(P_lhs, make_proxy(tmp));
    }
  else
    {
    coot_rt_t::copy(P_lhs, P_glue);
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator-=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(*this);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);

  // We will create the eGlue Proxy manually so as to not make P_lhs twice...
  // but in order to do that, we need to ensure that we have the right type for the Op proxy.
  typedef typename Proxy_glue_type< Op<T1, op_repmat>, subview_each2<parent, mode, TB> >::result Proxy_op_type;
  const Proxy< Proxy_op_type > P_op(op);
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T1, op_repmat>, eglue_minus>> P_glue(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());
  if (P_glue.is_empty())
    {
    return;
    }

  if (P_op.is_alias(subview_each_common<parent, mode>::P))
    {
    // alias_wrapper does not set the size correctly for a subview_elem2, so create the temporary matrix manually.
    Mat<typename parent::elem_type> tmp(P_op.get_n_rows(), P_op.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P_glue);
    coot_rt_t::copy(P_lhs, make_proxy(tmp));
    }
  else
    {
    coot_rt_t::copy(P_lhs, P_glue);
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator%=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(*this);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);

  // We will create the eGlue Proxy manually so as to not make P_lhs twice...
  // but in order to do that, we need to ensure that we have the right type for the Op proxy.
  typedef typename Proxy_glue_type< Op<T1, op_repmat>, subview_each2<parent, mode, TB> >::result Proxy_op_type;
  const Proxy< Proxy_op_type > P_op(op);
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T1, op_repmat>, eglue_schur>> P_glue(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());
  if (P_glue.is_empty())
    {
    return;
    }

  if (P_op.is_alias(subview_each_common<parent, mode>::P))
    {
    // alias_wrapper does not set the size correctly for a subview_elem2, so create the temporary matrix manually.
    Mat<typename parent::elem_type> tmp(P_op.get_n_rows(), P_op.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P_glue);
    coot_rt_t::copy(P_lhs, make_proxy(tmp));
    }
  else
    {
    coot_rt_t::copy(P_lhs, P_glue);
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator/=(const Base<elem_type, T1>& in)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(*this);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(in.get_ref(), copies_per_row, copies_per_col);

  // We will create the eGlue Proxy manually so as to not make P_lhs twice...
  // but in order to do that, we need to ensure that we have the right type for the Op proxy.
  typedef typename Proxy_glue_type< Op<T1, op_repmat>, subview_each2<parent, mode, TB> >::result Proxy_op_type;
  const Proxy< Proxy_op_type > P_op(op);
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T1, op_repmat>, eglue_div>> P_glue(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());
  if (P_glue.is_empty())
    {
    return;
    }

  if (P_op.is_alias(subview_each_common<parent, mode>::P))
    {
    // alias_wrapper does not set the size correctly for a subview_elem2, so create the temporary matrix manually.
    Mat<typename parent::elem_type> tmp(P_op.get_n_rows(), P_op.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P_glue);
    coot_rt_t::copy(P_lhs, make_proxy(tmp));
    }
  else
    {
    coot_rt_t::copy(P_lhs, P_glue);
    }
  }



//
// subview_each1_aux
//



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_plus(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? X.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? X.P.n_cols : 1;

  parent& P_ref = access::rw(X.P);

  Mat<typename parent::elem_type> out;
  out.set_size(X.P.n_rows, X.P.n_cols);

  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T2, op_repmat>, eglue_plus> glue(P_ref, op);
  const Proxy<eGlue<parent, Op<T2, op_repmat>, eglue_plus>> P(glue);

  X.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_minus(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? X.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? X.P.n_cols : 1;

  parent& P_ref = access::rw(X.P);

  Mat<typename parent::elem_type> out;
  out.set_size(X.P.n_rows, X.P.n_cols);

  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T2, op_repmat>, eglue_minus> glue(P_ref, op);
  const Proxy<eGlue<parent, Op<T2, op_repmat>, eglue_minus>> P(glue);

  X.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename T1, typename parent, unsigned int mode>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_minus(const Base<typename parent::elem_type, T1>& X, const subview_each1<parent, mode>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? Y.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? Y.P.n_cols : 1;

  parent& P_ref = access::rw(Y.P);

  Mat<typename parent::elem_type> out;
  out.set_size(Y.P.n_rows, Y.P.n_cols);

  const Op<T1, op_repmat> op(X.get_ref(), copies_per_row, copies_per_col);
  const eGlue<Op<T1, op_repmat>, parent, eglue_minus> glue(op, P_ref);
  const Proxy<eGlue<Op<T1, op_repmat>, parent, eglue_minus>> P(glue);

  Y.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P1.P.get_n_rows(), P.P1.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_schur(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? X.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? X.P.n_cols : 1;

  parent& P_ref = access::rw(X.P);

  Mat<typename parent::elem_type> out;
  out.set_size(X.P.n_rows, X.P.n_cols);

  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T2, op_repmat>, eglue_schur> glue(P_ref, op);
  const Proxy<eGlue<parent, Op<T2, op_repmat>, eglue_schur>> P(glue);

  X.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_div(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? X.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? X.P.n_cols : 1;

  parent& P_ref = access::rw(X.P);

  Mat<typename parent::elem_type> out;
  out.set_size(X.P.n_rows, X.P.n_cols);

  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const eGlue<parent, Op<T2, op_repmat>, eglue_div> glue(P_ref, op);
  const Proxy<eGlue<parent, Op<T2, op_repmat>, eglue_div>> P(glue);

  X.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P2.P.get_n_rows(), P.P2.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename T1, typename parent, unsigned int mode>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_div(const Base<typename parent::elem_type, T1>& X, const subview_each1<parent, mode>& Y)
  {
  coot_debug_sigprint();

  const uword copies_per_row = (mode == 1) ? Y.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? Y.P.n_cols : 1;

  parent& P_ref = access::rw(Y.P);

  Mat<typename parent::elem_type> out;
  out.set_size(Y.P.n_rows, Y.P.n_cols);

  const Op<T1, op_repmat> op(X.get_ref(), copies_per_row, copies_per_col);
  const eGlue<Op<T1, op_repmat>, parent, eglue_div> glue(op, P_ref);
  const Proxy<eGlue<Op<T1, op_repmat>, parent, eglue_div>> P(glue);

  Y.check_size(P_ref.n_rows, P_ref.n_cols, copies_per_row, copies_per_col, P.P1.P.get_n_rows(), P.P1.P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);

  return out;
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_plus(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(X);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T2, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T2, op_repmat>, eglue_plus>> P(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_minus(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(X);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T2, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T2, op_repmat>, eglue_minus>> P(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }



template<typename T1, typename parent, unsigned int mode, typename TB>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_minus(const Base<typename parent::elem_type, T1>& X, const subview_each2<parent, mode, TB>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the RHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_rhs(Y);

  // Convert the LHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_rhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_rhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(X.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T1, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<Op<T1, op_repmat>, subview_each2<parent, mode, TB>, eglue_minus>> P(P_op, P_rhs);

  subview_each_common<parent, mode>::check_size(P_rhs.get_n_rows(), P_rhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_schur(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(X);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T2, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T2, op_repmat>, eglue_schur>> P(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_div(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the LHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_lhs(X);

  // Convert the RHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_lhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_lhs.get_n_cols() : 1;
  const Op<T2, op_repmat> op(Y.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T2, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<subview_each2<parent, mode, TB>, Op<T2, op_repmat>, eglue_div>> P(P_lhs, P_op);

  subview_each_common<parent, mode>::check_size(P_lhs.get_n_rows(), P_lhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }



template<typename T1, typename parent, unsigned int mode, typename TB>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_div(const Base<typename parent::elem_type, T1>& X, const subview_each2<parent, mode, TB>& Y)
  {
  coot_debug_sigprint();

  // Make a proxy for the RHS so we can compute the necessary number of copies.
  const Proxy<subview_each2<parent, mode, TB>> P_rhs(Y);

  // Convert the LHS into a repmat operation that broadcasts it to the right size.
  const uword copies_per_row = (mode == 1) ? P_rhs.get_n_rows() : 1;
  const uword copies_per_col = (mode == 0) ? P_rhs.get_n_cols() : 1;
  const Op<T1, op_repmat> op(X.get_ref(), copies_per_row, copies_per_col);
  const Proxy<Op<T1, op_repmat>> P_op(op);

  // Create the eGlue Proxy manually so as to not make P_lhs twice.
  const Proxy<eGlue<Op<T1, op_repmat>, subview_each2<parent, mode, TB>, eglue_div>> P(P_op, P_rhs);

  subview_each_common<parent, mode>::check_size(P_rhs.get_n_rows(), P_rhs.get_n_cols(), copies_per_row, copies_per_col, P_op.P.get_n_rows(), P_op.P.get_n_cols());

  Mat<typename parent::elem_type> out;
  out.set_size(P.get_n_rows(), P.get_n_cols());
  if (P.is_empty())
    {
    return out;
    }

  coot_rt_t::copy(make_proxy(out), P);
  return out;
  }
