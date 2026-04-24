// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



template<typename eT, typename T1>
inline
subview_elem1<eT,T1>::~subview_elem1()
  {
  coot_debug_sigprint();
  }


template<typename eT, typename T1>
coot_inline
subview_elem1<eT,T1>::subview_elem1(const Mat<eT>& in_m, const Base<uword,T1>& in_a)
  : m(in_m)
  , a(in_a)
  {
  coot_debug_sigprint();
  }



template<typename eT, typename T1>
coot_inline
subview_elem1<eT,T1>::subview_elem1(const Cube<eT>& in_q, const Base<uword,T1>& in_a)
  : fake_m( in_q.get_dev_mem(false), in_q.n_elem, 1 )
  ,      m( fake_m )
  ,      a( in_a   )
  {
  coot_debug_sigprint();
  }



template<typename eT, typename T1>
coot_inline
const Op<subview_elem1<eT,T1>,op_htrans>
subview_elem1<eT,T1>::t() const
  {
  return Op<subview_elem1<eT,T1>,op_htrans>(*this);
  }



template<typename eT, typename T1>
coot_inline
const Op<subview_elem1<eT,T1>,op_htrans>
subview_elem1<eT,T1>::ht() const
  {
  return Op<subview_elem1<eT,T1>,op_htrans>(*this);
  }



template<typename eT, typename T1>
coot_inline
const Op<subview_elem1<eT,T1>,op_strans>
subview_elem1<eT,T1>::st() const
  {
  return Op<subview_elem1<eT,T1>,op_strans>(*this);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::replace(const eT old_val, const eT new_val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem1<eT, T1>, eop_replace> E(*this, 'j', old_val, new_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



//template<typename eT, typename T1>
//inline
//void
//subview_elem1<eT,T1>::clean(const pod_type threshold)
//  {
//  coot_debug_sigprint();
//
//  Mat<eT> tmp(*this);
//
//  tmp.clean(threshold);
//
//  (*this).operator=(tmp);
//  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::clamp(const eT min_val, const eT max_val)
  {
  coot_debug_sigprint();

  coot_conform_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const eOp<subview_elem1<eT, T1>, eop_clamp> E(*this, 'j', min_val, max_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::fill(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::fill(make_proxy(*this), val);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::zeros()
  {
  coot_debug_sigprint();

  fill(eT(0));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::ones()
  {
  coot_debug_sigprint();

  fill(eT(1));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randu()
  {
  coot_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U(a.get_ref());
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_conform_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );

  Col<eT> tmp(E.M.n_elem, fill::randu);
  (*this).operator=(tmp);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randn()
  {
  coot_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U(a.get_ref());
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_conform_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );

  Col<eT> tmp(E.M.n_elem, fill::randn);
  (*this).operator=(tmp);
  }



template<typename eT, typename T1>
inline
bool
subview_elem1<eT,T1>::is_empty() const
  {
  SizeProxy<T1> S(a.get_ref());
  return (S.get_n_elem() == 0);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator+= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem1<eT, T1>, eop_scalar_plus> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator-= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem1<eT, T1>, eop_scalar_minus_post> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator*= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem1<eT, T1>, eop_scalar_times> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator/= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem1<eT, T1>, eop_scalar_div_post> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



//
//



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T2>& x)
  {
  coot_debug_sigprint();

  const Proxy<subview_elem1<eT, T1>> P_out(*this);
  const Proxy<subview_elem1<eT, T2>> P_in(x);

  coot_assert_same_size(P_out.get_n_rows(), P_out.get_n_cols(), P_in.get_n_rows(), P_in.get_n_cols(), "Mat::elem()");

  inexact_alias_wrapper<Mat<eT>, Proxy<subview_elem1<eT, T2>>> A((*this).m, P_in);
  if (A.using_aux)
    {
    coot_rt_t::copy(A.use, P_in);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.use;
    }
  else
    {
    coot_rt_t::copy(P_out, P_in);
    }
  }



//! work around compiler bugs
template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T1>& x)
  {
  coot_debug_sigprint();

  const Proxy<subview_elem1<eT, T1>> P_out(*this);
  const Proxy<subview_elem1<eT, T1>> P_in(x);

  coot_assert_same_size(P_out.get_n_rows(), P_out.get_n_cols(), P_in.get_n_rows(), P_in.get_n_cols(), "Mat::elem()");

  inexact_alias_wrapper<Mat<eT>, Proxy<subview_elem1<eT, T1>>> A(access::rw((*this).m), P_in);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.use), P_in);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.use;
    }
  else
    {
    coot_rt_t::copy(P_out, P_in);
    }
  }



template<typename eT, typename T1>
template<typename eglue_type, typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const subview_elem1<eT,T2>& x, const char* op_name)
  {
  const eGlue<subview_elem1<eT, T1>, subview_elem1<eT, T2>, eglue_type> G(*this, x);
  const Proxy<eGlue<subview_elem1<eT, T1>, subview_elem1<eT, T2>, eglue_type>> P(G);

  coot_assert_same_size(P.P1.get_n_rows(), P.P1.get_n_cols(), P.P2.get_n_rows(), P.P2.get_n_cols(), op_name);

  inexact_alias_wrapper<Mat<eT>, Proxy<subview_elem1<eT, T2>>> A(access::rw((*this).m), P.P2);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.use), P);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.use;
    }
  else
    {
    coot_rt_t::copy(P.P1, P);
    }
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const subview_elem1<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_plus>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const subview_elem1<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_minus>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const subview_elem1<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_schur>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const subview_elem1<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_div>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const Base<eT,T2>& x)
  {
  coot_debug_sigprint();

  const Proxy<subview_elem1<eT, T1>> P_out(*this);
  const Proxy<T2> P_in(x.get_ref());

  coot_assert_same_size(P_out.get_n_rows(), P_out.get_n_cols(), P_in.get_n_rows(), P_in.get_n_cols(), "Mat::elem()");

  inexact_alias_wrapper<Mat<eT>, Proxy<T2>> A(access::rw((*this).m), P_in);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.use), P_in);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.use;
    }
  else
    {
    coot_rt_t::copy(P_out, P_in);
    }
  }



template<typename eT, typename T1>
template<typename eglue_type, typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const Base<eT,T2>& x, const char* op_name)
  {
  const eGlue<subview_elem1<eT, T1>, T2, eglue_type> G(*this, x.get_ref());
  const Proxy<eGlue<subview_elem1<eT, T1>, T2, eglue_type>> P(G);

  coot_assert_same_size(P.P1.get_n_rows(), P.P1.get_n_cols(), P.P2.get_n_rows(), P.P2.get_n_cols(), op_name);

  inexact_alias_wrapper<Mat<eT>, Proxy<T2>> A(access::rw((*this).m), P.P2);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.use), P);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.use;
    }
  else
    {
    coot_rt_t::copy(P.P1, P);
    }
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const Base<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_plus>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const Base<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_minus>(x, "Mat::elem()::operator-=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const Base<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_schur>(x, "Mat::elem()::operator%=");
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const Base<eT,T2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_div>(x, "Mat::elem()::operator/=");
  }



//
//



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::extract(Mat<eT>& actual_out, const subview_elem1<eT,T1>& in)
  {
  coot_debug_sigprint();

  alias_wrapper<Mat<eT>, Mat<eT>, T1> W(actual_out, in.m, in.a.get_ref());

  Proxy<subview_elem1<eT, T1>> P(in);
  W.use.set_size(P.get_n_elem(), 1);

  coot_rt_t::copy(make_proxy_col(W.use), P);
  }
