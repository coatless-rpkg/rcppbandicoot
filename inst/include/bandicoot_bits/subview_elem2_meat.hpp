// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://ratml.org)
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


template<typename eT, typename sve2_type>
inline
subview_elem2<eT, sve2_type>::~subview_elem2()
  {
  coot_debug_sigprint();
  }


template<typename eT, typename sve2_type>
template<typename T1, typename T2>
coot_inline
subview_elem2<eT, sve2_type>::subview_elem2
  (
  const Mat<eT>&        in_m,
  const Base<uword,T1>& in_ri,
  const Base<uword,T2>& in_ci
  )
  : m(in_m        )
  , r(in_ri, in_ci)
  {
  coot_debug_sigprint();
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::randu()
  {
  r.randu(*this);
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::randn()
  {
  r.randn(*this);
  }



//
//



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::replace(const eT old_val, const eT new_val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem2<eT, sve2_type>, eop_replace> E(*this, 'j', old_val, new_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



//template<typename eT, typename sve2_type>
//inline
//void
//subview_elem2<eT, sve2_type>::clean(const pod_type threshold)
//  {
//  coot_debug_sigprint();
//
//  Mat<eT> tmp(*this);
//
//  tmp.clean(threshold);
//
//  (*this).operator=(tmp);
//  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::clamp(const eT min_val, const eT max_val)
  {
  coot_debug_sigprint();

  coot_conform_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const eOp<subview_elem2<eT, sve2_type>, eop_clamp> E(*this, 'j', min_val, max_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::fill(const eT val)
  {
  coot_debug_sigprint();

  // Creation of this Proxy will perform bounds checks.
  coot_rt_t::fill(make_proxy(*this), val);
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::zeros()
  {
  coot_debug_sigprint();

  (*this).fill(eT(0));
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::ones()
  {
  coot_debug_sigprint();

  (*this).fill(eT(1));
  }



template<typename eT, typename sve2_type>
inline
bool
subview_elem2<eT, sve2_type>::is_empty() const
  {
  return r.is_empty(*this);
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::operator+= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem2<eT, sve2_type>, eop_scalar_plus> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::operator-= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem2<eT, sve2_type>, eop_scalar_minus_post> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::operator*= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem2<eT, sve2_type>, eop_scalar_times> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::operator/= (const eT val)
  {
  coot_debug_sigprint();

  const eOp<subview_elem2<eT, sve2_type>, eop_scalar_div_post> E(*this, val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



//
//



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::inplace_eq(const subview_elem2<eT, sve2_type2>& x)
  {
  const Proxy<subview_elem2<eT, sve2_type>> P_out(*this);
  const Proxy<subview_elem2<eT, sve2_type2>> P_in(x);

  coot_assert_same_size(P_out.get_n_rows(), P_out.get_n_cols(), P_in.get_n_rows(), P_in.get_n_cols(), "Mat::elem()");

  inexact_alias_wrapper<Mat<eT>, Proxy<subview_elem2<eT, sve2_type2>>> A(access::rw((*this).m), P_in);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P_in);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(P_out, P_in);
    }
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::inplace_eq(const Base<eT, expr>& x)
  {
  const Proxy<subview_elem2<eT, sve2_type>> P_out(*this);
  const Proxy<expr> P_in(x.get_ref());

  coot_assert_same_size(P_out.get_n_rows(), P_out.get_n_cols(), P_in.get_n_rows(), P_in.get_n_cols(), "Mat::elem()");

  inexact_alias_wrapper<Mat<eT>, Proxy<expr>> A(access::rw((*this).m), P_in);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P_in);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(P_out, P_in);
    }
  }



template<typename eT, typename sve2_type>
template<typename eglue_type, typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::inplace_op(const subview_elem2<eT, sve2_type2>& x, const char* op_name)
  {
  const eGlue<subview_elem2<eT, sve2_type>, subview_elem2<eT, sve2_type2>, eglue_type> G(*this, x);
  const Proxy<eGlue<subview_elem2<eT, sve2_type>, subview_elem2<eT, sve2_type2>, eglue_type>> P(G);

  coot_assert_same_size(P.P1.get_n_rows(), P.P1.get_n_cols(), P.P2.get_n_rows(), P.P2.get_n_cols(), op_name);

  inexact_alias_wrapper<Mat<eT>, Proxy<subview_elem2<eT, sve2_type2>>> A(access::rw((*this).m), P.P2);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(P.P1, P);
    }
  }



template<typename eT, typename sve2_type>
template<typename eglue_type, typename expr>
inline
void
subview_elem2<eT, sve2_type>::inplace_op(const Base<eT, expr>& x, const char* op_name)
  {
  const eGlue<subview_elem2<eT, sve2_type>, expr, eglue_type> G(*this, x.get_ref());
  const Proxy<eGlue<subview_elem2<eT, sve2_type>, expr, eglue_type>> P(G);

  coot_assert_same_size(P.P1.get_n_rows(), P.P1.get_n_cols(), P.P2.get_n_rows(), P.P2.get_n_cols(), op_name);

  inexact_alias_wrapper<Mat<eT>, Proxy<expr>> A(access::rw((*this).m), P.P2);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    // special handling: disable copy after inexact_alias_wrapper deallocation and do it ourselves
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(P.P1, P);
    }
  }



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::operator= (const subview_elem2<eT, sve2_type2>& x)
  {
  coot_debug_sigprint();

  inplace_eq(x);
  }



// ! work around compiler bugs
template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::operator= (const subview_elem2<eT, sve2_type>& x)
  {
  coot_debug_sigprint();

  inplace_eq(x);
  }



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::operator+= (const subview_elem2<eT, sve2_type2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_plus>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::operator-= (const subview_elem2<eT, sve2_type2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_minus>(x, "Mat::elem()::operator-=");
  }



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::operator%= (const subview_elem2<eT, sve2_type2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_schur>(x, "Mat::elem()::operator%=");
  }



template<typename eT, typename sve2_type>
template<typename sve2_type2>
inline
void
subview_elem2<eT, sve2_type>::operator/= (const subview_elem2<eT, sve2_type2>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_div>(x, "Mat::elem()::operator/=");
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::operator= (const Base<eT,expr>& x)
  {
  coot_debug_sigprint();

  inplace_eq(x);
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::operator+= (const Base<eT,expr>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_plus>(x, "Mat::elem()::operator+=");
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::operator-= (const Base<eT,expr>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_minus>(x, "Mat::elem()::operator-=");
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::operator%= (const Base<eT,expr>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_schur>(x, "Mat::elem()::operator%=");
  }



template<typename eT, typename sve2_type>
template<typename expr>
inline
void
subview_elem2<eT, sve2_type>::operator/= (const Base<eT,expr>& x)
  {
  coot_debug_sigprint();

  inplace_op<eglue_div>(x, "Mat::elem()::operator/=");
  }



//
//



template<typename eT, typename sve2_type>
inline
void
subview_elem2<eT, sve2_type>::extract(Mat<eT>& actual_out, const subview_elem2<eT, sve2_type>& in)
  {
  coot_debug_sigprint();

  const Proxy<subview_elem2<eT, sve2_type>> P(in);

  if (P.is_alias(actual_out))
    {
    Mat<eT> tmp;
    tmp.set_size(P.get_n_rows(), P.get_n_cols());
    coot_rt_t::copy(make_proxy(tmp), P);
    actual_out.steal_mem(tmp);
    }
  else
    {
    actual_out.set_size(P.get_n_rows(), P.get_n_cols());
    coot_rt_t::copy(make_proxy(actual_out), P);
    }
  }



//
// subview_elem2_both
//



template<typename eT, typename T1, typename T2>
coot_inline
subview_elem2_both<eT, T1, T2>::subview_elem2_both(const Base<uword, T1>& in_ri, const Base<uword, T2>& in_ci)
  : base_ri(in_ri)
  , base_ci(in_ci)
  { }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2_both<eT, T1, T2>::randu(subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& base)
  {
  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  coot_debug_sigprint();

  unwrap<T1> U1(base_ri.get_ref());
  unwrap<T2> U2(base_ci.get_ref());

  extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
  extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  coot_conform_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
  coot_conform_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

  Mat<eT> tmp(E1.M.n_elem, E2.M.n_elem, fill::randu);
  base = tmp;
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2_both<eT, T1, T2>::randn(subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& base)
  {
  coot_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U1(base_ri.get_ref());
  unwrap<T2> U2(base_ci.get_ref());

  extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
  extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  coot_conform_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
  coot_conform_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

  Mat<eT> tmp(E1.M.n_elem, E2.M.n_elem, fill::randn);
  base = tmp;
  }



template<typename eT, typename T1, typename T2>
inline
bool
subview_elem2_both<eT, T1, T2>::is_empty(const subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& s) const
  {
  coot_debug_sigprint();

  SizeProxy<T1> S1(base_ri.get_ref());
  SizeProxy<T2> S2(base_ci.get_ref());
  return (S1.get_n_elem() == 0) || (S2.get_n_elem() == 0);
  }



//
// subview_elem2_all_cols
//



template<typename eT, typename T1>
template<typename T2>
coot_inline
subview_elem2_all_cols<eT, T1>::subview_elem2_all_cols(const Base<uword, T1>& in_ri, const Base<uword, T2>& in_ci)
  : base_ri(in_ri)
  {
  coot_ignore(in_ci);
  }



template<typename eT, typename T1>
inline
void
subview_elem2_all_cols<eT, T1>::randu(subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& base)
  {
  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  coot_debug_sigprint();

  unwrap<T1> U1(base_ri.get_ref());

  extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

  coot_conform_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

  Mat<eT> tmp(E1.M.n_elem, base.m.n_cols, fill::randu);
  base = tmp;
  }



template<typename eT, typename T1>
inline
void
subview_elem2_all_cols<eT, T1>::randn(subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& base)
  {
  coot_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U1(base_ri.get_ref());

  extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

  coot_conform_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

  Mat<eT> tmp(E1.M.n_elem, base.m.n_cols, fill::randn);
  base = tmp;
  }



template<typename eT, typename T1>
inline
bool
subview_elem2_all_cols<eT, T1>::is_empty(const subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& s) const
  {
  coot_debug_sigprint();

  SizeProxy<T1> S(base_ri.get_ref());
  return (S.get_n_elem() == 0) || (s.m.n_cols == 0);
  }



//
// subview_elem2_all_rows
//



template<typename eT, typename T2>
template<typename T1>
coot_inline
subview_elem2_all_rows<eT, T2>::subview_elem2_all_rows(const Base<uword, T1>& in_ri, const Base<uword, T2>& in_ci)
  : base_ci(in_ci)
  {
  coot_ignore(in_ri);
  }



template<typename eT, typename T2>
inline
void
subview_elem2_all_rows<eT, T2>::randu(subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& base)
  {
  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  coot_debug_sigprint();

  unwrap<T2> U2(base_ci.get_ref());

  extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  coot_conform_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

  Mat<eT> tmp(base.m.n_rows, E2.M.n_elem, fill::randu);
  base = tmp;
  }



template<typename eT, typename T2>
inline
void
subview_elem2_all_rows<eT, T2>::randn(subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& base)
  {
  coot_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T2> U2(base_ci.get_ref());

  extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  coot_conform_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

  Mat<eT> tmp(base.m.n_rows, E2.M.n_elem, fill::randn);
  base = tmp;
  }



template<typename eT, typename T2>
inline
bool
subview_elem2_all_rows<eT, T2>::is_empty(const subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& s) const
  {
  coot_debug_sigprint();

  SizeProxy<T2> S(base_ci.get_ref());
  return (s.m.n_rows == 0) || (S.get_n_elem() == 0);
  }
