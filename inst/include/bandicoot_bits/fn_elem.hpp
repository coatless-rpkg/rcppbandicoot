// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
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



//
// square

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_square> >::result
square(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_square>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_square>
square(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_square>(A.get_ref());
  }



//
// sqrt

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sqrt> >::result
sqrt(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sqrt>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_sqrt>
sqrt(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_sqrt>(A.get_ref());
  }



//
// exp

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp> >::result
exp(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_exp>
exp(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_exp>(A.get_ref());
  }



//
// exp2

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp2> >::result
exp2(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp2>(A);
  }




template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_exp2>
exp2(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_exp2>(A.get_ref());
  }



//
// exp10

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp10> >::result
exp10(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp10>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_exp10>
exp10(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_exp10>(A.get_ref());
  }



//
// trunc_exp

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_trunc_exp> >::result
trunc_exp(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_trunc_exp>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_trunc_exp>
trunc_exp(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_trunc_exp>(A.get_ref());
  }



//
// log

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log> >::result
log(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_log>
log(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_log>(A.get_ref());
  }



//
// log2

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log2> >::result
log2(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log2>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_log2>
log2(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_log2>(A.get_ref());
  }



//
// log10

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log10> >::result
log10(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log10>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_log10>
log10(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_log10>(A.get_ref());
  }



//
// trunc_log

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_trunc_log> >::result
trunc_log(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_trunc_log>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_trunc_log>
trunc_log(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_trunc_log>(A.get_ref());
  }



//
// cos

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_cos> >::result
cos(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_cos>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_cos>
cos(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_cos>(A.get_ref());
  }



//
// sin

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sin> >::result
sin(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sin>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_sin>
sin(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_sin>(A.get_ref());
  }



//
// tan

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_tan> >::result
tan(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_tan>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_tan>
tan(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_tan>(A.get_ref());
  }



//
// acos

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_acos> >::result
acos(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_acos>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_acos>
acos(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_acos>(A.get_ref());
  }



//
// asin

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_asin> >::result
asin(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_asin>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_asin>
asin(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_asin>(A.get_ref());
  }



//
// atan

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_atan> >::result
atan(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_atan>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_atan>
atan(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_atan>(A.get_ref());
  }



//
// cosh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_cosh> >::result
cosh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_cosh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_cosh>
cosh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_cosh>(A.get_ref());
  }



//
// sinh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sinh> >::result
sinh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sinh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_sinh>
sinh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_sinh>(A.get_ref());
  }



//
// tanh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_tanh> >::result
tanh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_tanh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_tanh>
tanh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_tanh>(A.get_ref());
  }



//
// acosh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_acosh> >::result
acosh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_acosh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_acosh>
acosh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_acosh>(A.get_ref());
  }



//
// asinh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_asinh> >::result
asinh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_asinh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_asinh>
asinh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_asinh>(A.get_ref());
  }



//
// atanh

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_atanh> >::result
atanh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_atanh>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_atanh>
atanh(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_atanh>(A.get_ref());
  }



//
// sinc

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sinc> >::result
sinc(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sinc>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_sinc>
sinc(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_sinc>(A.get_ref());
  }



//
// atan2

template<typename T1, typename T2>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_coot_type<T2>::value && is_real<typename T1::elem_type>::value && is_real<typename T2::elem_type>::value, const eGlue<T1, T2, eglue_atan2> >::result
atan2(const T1& X, const T2& Y)
  {
  coot_extra_debug_sigprint();

  return eGlue<T1, T2, eglue_atan2>(X, Y);
  }

// TODO: cube version



//
// hypot

template<typename T1, typename T2>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_coot_type<T2>::value && is_real<typename T1::elem_type>::value && is_real<typename T2::elem_type>::value, const eGlue<T1, T2, eglue_hypot> >::result
hypot(const T1& X, const T2& Y)
  {
  coot_extra_debug_sigprint();

  return eGlue<T1, T2, eglue_hypot>(X, Y);
  }

// TODO: cube version



//
// abs

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && std::is_signed<typename T1::elem_type>::value, const eOp<T1, eop_abs> >::result
abs(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_abs>(A);
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< std::is_signed<typename T1::elem_type>::value, const eOpCube<T1, eop_abs> >::result
abs(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_abs>(A.get_ref());
  }



// abs(unsigned)... nothing to do
template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && !std::is_signed<typename T1::elem_type>::value, const T1&>::result
abs(const T1& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< !std::is_signed<typename T1::elem_type>::value, const T1&>::result
abs(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return A.get_ref();
  }



// abs(abs)... nothing to do
template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_abs>& >::result
abs(const eOp<T1, eop_abs>& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_cube_type<T1>::value, const eOpCube<T1, eop_abs>& >::result
abs(const eOpCube<T1, eop_abs>& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



//
// pow

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_pow> >::result
pow(const T1& A, const typename T1::elem_type exponent)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_pow>(A, exponent);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_pow>
pow(const BaseCube<typename T1::elem_type, T1>& A, const typename T1::elem_type exponent)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_pow>(A.get_ref(), exponent);
  }



//
// floor
// TODO: optimizations to skip processing entirely for integer types

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_floor> >::result
floor(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_floor>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_floor>
floor(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_floor>(A.get_ref());
  }



//
// ceil

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_ceil> >::result
ceil(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_ceil>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_ceil>
ceil(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_ceil>(A.get_ref());
  }



//
// round

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_round> >::result
round(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_round>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_round>
round(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_round>(A.get_ref());
  }



//
// trunc

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_trunc> >::result
trunc(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_trunc>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_trunc>
trunc(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_trunc>(A.get_ref());
  }



//
// sign

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_sign> >::result
sign(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sign>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_sign>
sign(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_sign>(A.get_ref());
  }



//
// erf

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_erf> >::result
erf(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_erf>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_erf>
erf(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_erf>(A.get_ref());
  }



//
// erfc

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_erfc> >::result
erfc(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_erfc>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_erfc>
erfc(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_erfc>(A.get_ref());
  }



//
// lgamma

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, eOp<T1, eop_lgamma> >::result
lgamma(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_lgamma>(A);
  }



template<typename T1>
coot_warn_unused
inline
const eOpCube<T1, eop_lgamma>
lgamma(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_lgamma>(A.get_ref());
  }



//
// real

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_cx<typename T1::elem_type>::no, const T1& >::result
real(const T1& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_cx<typename T1::elem_type>::no, const BaseCube<typename T1::elem_type, T1>& >::result
real(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



// NOTE: complex elements are not officially supported!  This will only work
// with the CUDA backend.
template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_cx<typename T1::elem_type>::yes, const mtOp<typename T1::pod_type, T1, mtop_real> >::result
real(const T1& A)
  {
  coot_extra_debug_sigprint();

  return mtOp<typename T1::pod_type, T1, mtop_real>(A);
  }


//
// imag

template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_cx<typename T1::elem_type>::no, Mat<typename T1::pod_type>& >::result
imag(const T1& A)
  {
  coot_extra_debug_sigprint();

  SizeProxy<T1> sp(A);

  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();

  return Mat<typename T1::pod_type>(n_rows, n_cols, fill::zeros);
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_cx<typename T1::elem_type>::no, Cube<typename T1::pod_type>& >::result
imag(const BaseCube<typename T1::elem_type, T1>& A)
  {
  coot_extra_debug_sigprint();

  SizeProxyCube<T1> sp(A);

  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();
  const uword n_slices = A.get_n_slices();

  return Cube<typename T1::pod_type>(n_rows, n_cols, n_slices, fill::zeros);
  }



// NOTE: complex elements are not officially supported!  This will only work
// with the CUDA backend.
template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value && is_cx<typename T1::elem_type>::yes, const mtOp<typename T1::pod_type, T1, mtop_imag> >::result
imag(const T1& A)
  {
  coot_extra_debug_sigprint();

  return mtOp<typename T1::pod_type, T1, mtop_imag>(A);
  }
