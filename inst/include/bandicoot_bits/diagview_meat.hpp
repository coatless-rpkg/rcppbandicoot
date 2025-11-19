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
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
coot_inline
diagview<eT>::diagview(const Mat<eT>& in_m, const uword in_row_offset, const uword in_col_offset, const uword in_len)
  : m         (in_m                                       )
  , mem_offset(in_row_offset + in_col_offset * in_m.n_rows)
  , n_rows    (in_len                                     )
  , n_elem    (in_len                                     )
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
diagview<eT>::diagview(const diagview<eT>& in)
  : m         (in.m         )
  , mem_offset(in.mem_offset)
  , n_rows    (in.n_rows    )
  , n_elem    (in.n_elem    )
  {
  coot_extra_debug_sigprint(coot_str::format("this = %x   in = %x") % this % &in);
  }



template<typename eT>
inline
diagview<eT>::diagview(diagview<eT>&& in)
  : m         (in.m         )
  , mem_offset(in.mem_offset)
  , n_rows    (in.n_rows    )
  , n_elem    (in.n_elem    )
  {
  coot_extra_debug_sigprint(coot_str::format("this = %x   in = %x") % this % &in);

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
  coot_extra_debug_sigprint();

  coot_debug_check( (n_elem != x.n_elem), "diagview: diagonals have incompatible lengths" );

        Mat<eT>& d_m = const_cast< Mat<eT>& >(m);
  const Mat<eT>& x_m = x.m;

  // We can view the diagonal as a subview.
  coot_rt_t::copy_mat(d_m.get_dev_mem(false), x_m.get_dev_mem(false),
                      1, n_elem,
                      mem_offset, 0, d_m.n_rows + 1,
                      x.mem_offset, 0, x_m.n_rows + 1);
  }



template<typename eT>
inline
void
diagview<eT>::operator+=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
                        m.get_dev_mem(false), m.get_dev_mem(false),
                        (eT) val, (eT) 0,
                        1, n_elem, 1,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols);
  }



template<typename eT>
inline
void
diagview<eT>::operator-=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_minus_scalar_post,
                        m.get_dev_mem(false), m.get_dev_mem(false),
                        (eT) val, (eT) 0,
                        1, n_elem, 1,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols);
  }



template<typename eT>
inline
void
diagview<eT>::operator*=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                        m.get_dev_mem(false), m.get_dev_mem(false),
                        (eT) val, (eT) 1,
                        1, n_elem, 1,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols);
  }



template<typename eT>
inline
void
diagview<eT>::operator/=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                        m.get_dev_mem(false), m.get_dev_mem(false),
                        (eT) val, (eT) 1,
                        1, n_elem, 1,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols,
                        mem_offset, 0, 0, m.n_rows + 1, m.n_cols);
  }



//! set a diagonal of our matrix using data from a foreign object
template<typename eT>
inline
void
diagview<eT>::operator= (const Mat<eT>& o)
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    ( (n_elem != o.n_elem) || ((o.n_rows != 1) && (o.n_cols != 1)) ),
    "diagview: given object has incompatible size"
    );

  alias_wrapper<diagview<eT>, Mat<eT>> W(*this, o);
  coot_rt_t::copy_mat(W.get_dev_mem(false), o.get_dev_mem(false),
                      1, n_elem,
                      W.get_row_offset(), 0, W.get_incr(),
                      0, 0, 1);
  }



template<typename eT>
inline
void
diagview<eT>::operator= (const subview<eT>& o)
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    ( (n_elem != o.n_elem) || ((o.n_rows != 1) && (o.n_cols != 1)) ),
    "diagview: given object has incompatible size"
    );

  const bool is_vector = (o.n_rows == 1 || o.n_cols == 1);
  alias_wrapper<diagview<eT>, subview<eT>> W(*this, o);
  coot_rt_t::copy_mat(W.get_dev_mem(false), o.m.get_dev_mem(false),
                      1, n_elem,
                      W.get_row_offset(), 0, W.get_incr(),
                      o.aux_row1, o.aux_col1, is_vector ? 1 : o.m.n_rows + 1);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator= (const Base<eT,T1>& o)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U( o.get_ref() );

  operator=(U.M);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator+=(const Base<eT,T1>& o)
  {
  coot_extra_debug_sigprint();

  // TODO: dedicated kernels for these in-place operations
  Mat<eT> tmp(*this);
  tmp += o.get_ref();
  operator=(tmp);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator-=(const Base<eT,T1>& o)
  {
  coot_extra_debug_sigprint();

  // TODO: dedicated kernels for these in-place operations
  Mat<eT> tmp(*this);
  tmp -= o.get_ref();
  operator=(tmp);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator%=(const Base<eT,T1>& o)
  {
  coot_extra_debug_sigprint();

  // TODO: dedicated kernels for these in-place operations
  Mat<eT> tmp(*this);
  tmp %= o.get_ref();
  operator=(tmp);
  }



template<typename eT>
template<typename T1>
inline
void
diagview<eT>::operator/=(const Base<eT,T1>& o)
  {
  coot_extra_debug_sigprint();

  // TODO: dedicated kernels for these in-place operations
  Mat<eT> tmp(*this);
  tmp /= o.get_ref();
  operator=(tmp);
  }



//! extract a diagonal and store it as a column vector
template<typename eT>
inline
void
diagview<eT>::extract(Mat<eT>& out, const diagview<eT>& in)
  {
  coot_extra_debug_sigprint();

  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()

  out.set_size(in.n_rows, in.n_cols); // should be a vector

  // A diagonal can be seen as a subvector of the matrix with m_n_rows = n_rows + 1.
  coot_rt_t::copy_mat(out.get_dev_mem(false), in.m.get_dev_mem(false),
                      1, in.n_elem,
                      0, 0, 1,
                      in.mem_offset, 0, in.m.n_rows + 1);
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
  coot_debug_check_bounds( (ii >= n_elem), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + ii * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::operator()(const uword ii) const
  {
  coot_debug_check_bounds( (ii >= n_elem), "diagview::operator(): out of bounds" );

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
  coot_debug_check_bounds( ((row >= n_elem) || (col > 0)), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + row * (m.n_rows + 1);
  return (const_cast< Mat<eT>& >(m)).at(index);
  }



template<typename eT>
inline
eT
diagview<eT>::operator()(const uword row, const uword col) const
  {
  coot_debug_check_bounds( ((row >= n_elem) || (col > 0)), "diagview::operator(): out of bounds" );

  const uword index = mem_offset + row * (m.n_rows + 1);
  return m.at(index);
  }



/* template<typename eT> */
/* inline */
/* void */
/* diagview<eT>::replace(const eT old_val, const eT new_val) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   Mat<eT>& x = const_cast< Mat<eT>& >(m); */

/*   const uword local_n_elem = n_elem; */

/*   if(arma_isnan(old_val)) */
/*     { */
/*     for(uword ii=0; ii < local_n_elem; ++ii) */
/*       { */
/*       eT& val = x.at(ii+row_offset, ii+col_offset); */

/*       val = (arma_isnan(val)) ? new_val : val; */
/*       } */
/*     } */
/*   else */
/*     { */
/*     for(uword ii=0; ii < local_n_elem; ++ii) */
/*       { */
/*       eT& val = x.at(ii+row_offset, ii+col_offset); */

/*       val = (val == old_val) ? new_val : val; */
/*       } */
/*     } */
/*   } */



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
  coot_extra_debug_sigprint();

  // TODO: a dedicated implementation could be possible
  Mat<eT> tmp(*this);

  tmp.clamp(min_val, max_val);

  (*this).operator=(tmp);
  }



template<typename eT>
inline
void
diagview<eT>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::fill(m.get_dev_mem(false), val,
                  1, n_elem,
                  mem_offset, 0, m.n_rows + 1);
  }



template<typename eT>
inline
void
diagview<eT>::zeros()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(0));
  }



template<typename eT>
inline
void
diagview<eT>::ones()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(1));
  }



template<typename eT>
inline
void
diagview<eT>::randu()
  {
  coot_extra_debug_sigprint();

  Col<eT> r;
  r.randu(n_elem);
  operator=(r);
  }



template<typename eT>
inline
void
diagview<eT>::randn()
  {
  coot_extra_debug_sigprint();

  Col<eT> r;
  r.randn(n_elem);
  operator=(r);
  }
