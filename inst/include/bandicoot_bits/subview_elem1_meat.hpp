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
  coot_extra_debug_sigprint();
  }


template<typename eT, typename T1>
coot_inline
subview_elem1<eT,T1>::subview_elem1(const Mat<eT>& in_m, const Base<uword,T1>& in_a)
  : m(in_m)
  , a(in_a)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT, typename T1>
coot_inline
subview_elem1<eT,T1>::subview_elem1(const Cube<eT>& in_q, const Base<uword,T1>& in_a)
  : fake_m( in_q.get_dev_mem(false), in_q.n_elem, 1 )
  ,      m( fake_m )
  ,      a( in_a   )
  {
  coot_extra_debug_sigprint();
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::inplace_op(const twoway_kernel_id::enum_id kernel_id, const eT val_pre, const eT val_post)
  {
  coot_extra_debug_sigprint();

  // For now we need to instantiate the actual indices we will use.
  const unwrap<T1> U(a.get_ref());
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );

  // A bounds check is tedious but the only way that we can give the user an error.
  coot_debug_check( E.M.max() >= m.n_elem, "Mat::elem(): index out of bounds" );

  alias_wrapper<Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E.M);

  coot_rt_t::eop_scalar_subview_elem1(kernel_id,
                                      A.get_dev_mem(false),
                                      E.M.get_dev_mem(false),
                                      m.get_dev_mem(false),
                                      E.M.get_dev_mem(false),
                                      val_pre, val_post,
                                      E.M.n_elem);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const twoway_kernel_id::enum_id kernel_id, const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  if(is_alias(m, x.m))
    {
    coot_extra_debug_print("subview_elem1::inplace_op(): aliasing detected");

    const Mat<eT> tmp(x);
    inplace_op(kernel_id, tmp);
    }
  else
    {
    // Make sure the results will be the same size.
    SizeProxy<T1> S1(a.get_ref());
    SizeProxy<T2> S2(x.a.get_ref());

    coot_debug_check( S1.get_n_rows() != 1 && S1.get_n_cols() != 1, "Mat::elem(): indices must be a vector" );
    coot_debug_check( S2.get_n_rows() != 1 && S2.get_n_cols() != 1, "Mat::elem(): indices must be a vector" );

    coot_debug_check( S1.get_n_elem() != S2.get_n_elem(), "Mat::elem(): size mismatch" );

    // We have to unwrap both index vectors.
    const unwrap<T1> U1(a.get_ref());
    const unwrap<T2> U2(x.a.get_ref());

    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.max() >= m.n_elem, "Mat::elem(): index out of bounds" );
    coot_debug_check( E2.M.max() >= x.m.n_elem, "Mat::elem(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), const_cast<Mat<eT>&>(x.m), E1.M, E2.M);

    coot_rt_t::eop_subview_elem1(kernel_id,
                                 A.get_dev_mem(false),
                                 E1.M.get_dev_mem(false),
                                 x.m.get_dev_mem(false),
                                 E2.M.get_dev_mem(false),
                                 E1.M.n_elem);
    }
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const twoway_kernel_id::enum_id kernel_id, const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U1(a.get_ref());
  const unwrap<T2> U2(x.get_ref());

  const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
  const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): indices must be a vector" );
  coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): argument must be a vector" );

  coot_debug_check( E1.M.n_elem != E2.M.n_elem, "Mat::elem(): size mismatch" );

  alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M);

  coot_rt_t::eop_subview_elem1_array(kernel_id,
                                     A.get_dev_mem(false),
                                     E1.M.get_dev_mem(false),
                                     E2.M.get_dev_mem(false),
                                     E1.M.n_elem);
  }



//
//



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
  coot_extra_debug_sigprint();

  // ugly, slow implementation until we have kernels that can handle any kind of subview addressing
  Mat<eT> tmp(*this);
  tmp.replace(old_val, new_val);
  (*this).operator=(tmp);
  }



//template<typename eT, typename T1>
//inline
//void
//subview_elem1<eT,T1>::clean(const pod_type threshold)
//  {
//  coot_extra_debug_sigprint();
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
  coot_extra_debug_sigprint();

  // ugly, slow implementation until we have kernels that can handle any kind of subview addressing
  Mat<eT> tmp(*this);
  tmp.clamp(min_val, max_val);
  (*this).operator=(tmp);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  // For now we need to instantiate the actual indices we will use.
  unwrap<T1> U(a.get_ref());
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  // A bounds check is tedious but the only way that we can give the user an error.
  coot_debug_check( E.M.max() >= m.n_elem, "Mat::elem(): index out of bounds" );

  coot_rt_t::fill_subview_elem1(m.get_dev_mem(false),
                                E.M.get_dev_mem(false),
                                val,
                                E.M.n_elem);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::zeros()
  {
  coot_extra_debug_sigprint();

  fill(eT(0));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::ones()
  {
  coot_extra_debug_sigprint();

  fill(eT(1));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randu()
  {
  coot_extra_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U(a.get_ref());
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );

  Col<eT> tmp(E.M.n_elem, fill::randu);
  (*this).operator=(tmp);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randn()
  {
  coot_extra_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  unwrap<T1> U(a.get_ref());
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );

  Col<eT> tmp(E.M.n_elem, fill::randn);
  (*this).operator=(tmp);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator+= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_plus_scalar_sve1, val, (eT) 0);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator-= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_minus_scalar_post_sve1, val, (eT) 0);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator*= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_mul_scalar_sve1, val, (eT) 1);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator/= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_div_scalar_post_sve1, val, (eT) 1);
  }



//
//



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_eq_sve1, x);
  }



//! work around compiler bugs
template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T1>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_eq_sve1, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_plus_sve1, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_minus_sve1, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_mul_sve1, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const subview_elem1<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_div_sve1, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_eq_array, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_plus_array, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_minus_array, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_mul_array, x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const Base<eT,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve1_div_array, x);
  }



//
//



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::extract(Mat<eT>& actual_out, const subview_elem1<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.a.get_ref());
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::elem(): indices must be a vector" );
  coot_debug_check( E.M.max() >= in.m.n_elem, "Mat::extract(): index out of bounds" );

  if (is_alias(actual_out, E.M) || is_alias(actual_out, in.m))
    {
    Mat<eT> tmp(E.M.n_elem, 1);

    coot_rt_t::extract_subview_elem1(tmp.get_dev_mem(false),
                                     in.m.get_dev_mem(false),
                                     E.M.get_dev_mem(false),
                                     E.M.n_elem);

    actual_out.steal_mem(tmp);
    }
  else
    {
    actual_out.set_size(E.M.n_elem, 1);

    coot_rt_t::extract_subview_elem1(actual_out.get_dev_mem(false),
                                     in.m.get_dev_mem(false),
                                     E.M.get_dev_mem(false),
                                     E.M.n_elem);
    }
  }
