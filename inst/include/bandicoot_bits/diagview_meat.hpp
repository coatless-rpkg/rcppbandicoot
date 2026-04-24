// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2022      Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename eT>
inline
diagview<eT>::~diagview()
  {
  coot_debug_sigprint_this(this);
  }



template<typename eT>
coot_inline
diagview<eT>::diagview(const Mat<eT>& in_m, const uword in_row_offset, const uword in_col_offset, const uword in_len)
  : m         (in_m                                       )
  , mem_offset(in_row_offset + in_col_offset * in_m.n_rows)
  , n_rows    (in_len                                     )
  , n_elem    (in_len                                     )
  {
  coot_debug_sigprint_this(this);
  }



template<typename eT>
inline
diagview<eT>::diagview(const diagview<eT>& in)
  : m         (in.m         )
  , mem_offset(in.mem_offset)
  , n_rows    (in.n_rows    )
  , n_elem    (in.n_elem    )
  {
  coot_debug_sigprint(coot_str::format("this = %x; in = %x") % this % &in);
  }



template<typename eT>
inline
diagview<eT>::diagview(diagview<eT>&& in)
  : m         (in.m         )
  , mem_offset(in.mem_offset)
  , n_rows    (in.n_rows    )
  , n_elem    (in.n_elem    )
  {
  coot_debug_sigprint(coot_str::format("this = %x; in = %x") % this % &in);

  // for paranoia

  access::rw(in.mem_offset) = 0;
  access::rw(in.n_rows    ) = 0;
  access::rw(in.n_elem    ) = 0;
  }



//! set a diagonal of our matrix using a diagonal from a foreign matrix
template<typename eT>
inline
void
diagview<eT>::operator= (const diagview<eT>& x)
  {
  coot_debug_sigprint();

  coot_conform_check( (n_elem != x.n_elem), "diagview: diagonals have incompatible lengths" );

        Mat<eT>& d_m = const_cast< Mat<eT>& >(m);
  const Mat<eT>& x_m = x.m;

  // We can view the diagonal as a subview.
  coot_rt_t::copy(make_proxy(*this), make_proxy(x));
  }



template<typename eT>
inline
void
diagview<eT>::operator+=(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<diagview<eT>, eop_scalar_plus>(*this, val)));
  }



template<typename eT>
inline
void
diagview<eT>::operator-=(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<diagview<eT>, eop_scalar_minus_post>(*this, val)));
  }



template<typename eT>
inline
void
diagview<eT>::operator*=(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<diagview<eT>, eop_scalar_times>(*this, val)));
  }



template<typename eT>
inline
void
diagview<eT>::operator/=(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<diagview<eT>, eop_scalar_div_post>(*this, val)));
  }



//! set a diagonal of our matrix using data from a foreign object
template<typename eT>
inline
void
diagview<eT>::operator= (const Mat<eT>& o)
  {
  coot_debug_sigprint();

  coot_conform_check
    (
    ( (n_elem != o.n_elem) || ((o.n_rows != 1) && (o.n_cols != 1)) ),
    "diagview: given object has incompatible size"
    );

  alias_wrapper<diagview<eT>, Mat<eT>> W(*this, o);
  if (W.using_aux)
    {
    coot_rt_t::copy(make_proxy_col(W.aux), make_proxy_col(o));
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), make_proxy_col(o));
    }
  }



template<typename eT>
inline
void
diagview<eT>::operator= (const subview<eT>& o)
  {
  coot_debug_sigprint();

  coot_conform_check
    (
    ( (n_elem != o.n_elem) || ((o.n_rows != 1) && (o.n_cols != 1)) ),
    "diagview: given object has incompatible size"
    );

  const bool is_vector = (o.n_rows == 1 || o.n_cols == 1);
  alias_wrapper<diagview<eT>, subview<eT>> W(*this, o);
  if (W.using_aux)
    {
    coot_rt_t::copy(make_proxy_col(W.aux), make_proxy_col(o));
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), make_proxy_col(o));
    }
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator= (const Base<eT,T1>& o)
  {
  coot_debug_sigprint();

  const Proxy<T1> P(o.get_ref());

  coot_conform_check
    (
    ( (n_elem != P.get_n_elem()) || ((P.get_n_rows() != 1) && (P.get_n_cols() != 1)) ),
    "diagview: given object has incompatible size"
    );

  alias_wrapper<diagview<eT>, Proxy<T1>> A(*this, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    A.using_aux = false;
    (*this) = A.aux;
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), P);
    }
  }



template<typename eT>
template<typename eglue_type, typename T1>
inline
void
diagview<eT>::inplace_op(const Base<eT,T1>& o)
  {
  const eGlue<diagview<eT>, T1, eglue_type> G(*this, o.get_ref());
  const Proxy<eGlue<diagview<eT>, T1, eglue_type>> P(G);

  coot_conform_check
    (
    ( (n_elem != P.get_n_elem()) || ((P.get_n_rows() != 1) && (P.get_n_cols() != 1)) ),
    "diagview: given object has incompatible size"
    );

  alias_wrapper<diagview<eT>, Proxy<T1>> A(*this, P.P2);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    // disable copy on deallocation and copy manually
    A.using_aux = false;
    (*this) = A.aux;
    }
  else
    {
    coot_rt_t::copy(P.P1, P);
    }
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator+=(const Base<eT,T1>& o)
  {
  coot_debug_sigprint();

  inplace_op<eglue_plus>(o);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator-=(const Base<eT,T1>& o)
  {
  coot_debug_sigprint();

  inplace_op<eglue_minus>(o);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator%=(const Base<eT,T1>& o)
  {
  coot_debug_sigprint();

  inplace_op<eglue_schur>(o);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator/=(const Base<eT,T1>& o)
  {
  coot_debug_sigprint();

  inplace_op<eglue_div>(o);
  }



//! extract a diagonal and store it as a column vector
template<typename eT>
inline
void
diagview<eT>::extract(Mat<eT>& out, const diagview<eT>& in)
  {
  coot_debug_sigprint();

  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()

  out.set_size(in.n_rows, in.n_cols); // should be a vector

  // A diagonal can be seen as a subvector of the matrix with m_n_rows = n_rows + 1.
  coot_rt_t::copy(make_proxy_col(out), make_proxy(in));
  }



template<typename eT>
inline
MatValProxy<eT>
diagview<eT>::operator[](const uword ii)
  {
  const uword index = mem_offset + ii * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::operator[](const uword ii) const
  {
  const uword index = mem_offset + ii * (m.n_rows + 1);
  return m.at(index);
  }



template<typename eT>
inline
MatValProxy<eT>
diagview<eT>::at(const uword ii)
  {
  const uword index = mem_offset + ii * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::at(const uword ii) const
  {
  const uword index = mem_offset + ii * (m.n_rows + 1);
  return m.at(index);
  }



template<typename eT>
inline
MatValProxy<eT>
diagview<eT>::operator()(const uword ii)
  {
  coot_conform_check_bounds( (ii >= n_elem), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + ii * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::operator()(const uword ii) const
  {
  coot_conform_check_bounds( (ii >= n_elem), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + ii * (m.n_rows + 1);
  return m.at(index);
  }



template<typename eT>
inline
MatValProxy<eT>
diagview<eT>::at(const uword row, const uword)
  {
  const uword index = mem_offset + row * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::at(const uword row, const uword) const
  {
  const uword index = mem_offset + row * (m.n_rows + 1);
  return m.at(index);
  }



template<typename eT>
inline
MatValProxy<eT>
diagview<eT>::operator()(const uword row, const uword col)
  {
  coot_conform_check_bounds( ((row >= n_elem) || (col > 0)), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + row * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::operator()(const uword row, const uword col) const
  {
  coot_conform_check_bounds( ((row >= n_elem) || (col > 0)), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + row * (m.n_rows + 1);
  return m.at(index);
  }



template<typename eT>
inline
void
diagview<eT>::replace(const eT old_val, const eT new_val)
  {
  coot_debug_sigprint();

  const eOp<diagview<eT>, eop_replace> E(*this, 'j', old_val, new_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



/* template<typename eT> */
/* inline */
/* void */
/* diagview<eT>::clean(const typename get_pod_type<eT>::result threshold) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   Mat<eT> tmp(*this); */

/*   tmp.clean(threshold); */

/*   (*this).operator=(tmp); */
/*   } */



template<typename eT>
inline
void
diagview<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_debug_sigprint();

  coot_conform_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const eOp<diagview<eT>, eop_clamp> E(*this, 'j', min_val, max_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT>
inline
void
diagview<eT>::fill(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::fill(make_proxy(*this), val);
  }



template<typename eT>
inline
void
diagview<eT>::zeros()
  {
  coot_debug_sigprint();

  (*this).fill(eT(0));
  }



template<typename eT>
inline
void
diagview<eT>::ones()
  {
  coot_debug_sigprint();

  (*this).fill(eT(1));
  }



template<typename eT>
inline
void
diagview<eT>::randu()
  {
  coot_debug_sigprint();

  Col<eT> r;
  r.randu(n_elem);
  operator=(r);
  }



template<typename eT>
inline
void
diagview<eT>::randn()
  {
  coot_debug_sigprint();

  Col<eT> r;
  r.randn(n_elem);
  operator=(r);
  }



template<typename eT>
inline
bool
diagview<eT>::is_empty() const
  {
  return (n_elem == 0);
  }
