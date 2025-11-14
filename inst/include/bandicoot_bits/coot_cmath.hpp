// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
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
// wrappers for isfinite


template<typename eT>
coot_inline
bool
coot_isfinite(const eT&)
  {
  return true;
  }



template<>
coot_inline
bool
coot_isfinite(const float& x)
  {
  return std::isfinite(x);
  }



template<>
coot_inline
bool
coot_isfinite(const double& x)
  {
  return std::isfinite(x);
  }



template<typename T>
coot_inline
bool
coot_isfinite(const std::complex<T>& x)
  {
  return ( coot_isfinite(x.real()) && coot_isfinite(x.imag()) );
  }



template<>
coot_inline
bool
coot_isfinite(const fp16_shim& x)
  {
  return coot_isfinite(x.x);
  }



#if defined(COOT_USE_CUDA)
template<>
coot_inline
bool
coot_isfinite(const __half& x)
  {
  #if CUDA_VERSION >= 12020
  return !__hisinf(x) && !__hisnan(x);
  #else
  return !coot_cuda_half_isinf(x) && !coot_cuda_half_isnan(x);
  #endif
  }
#endif



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<>
coot_inline
bool
coot_isfinite(const std::float16_t& x)
  {
  return std::isfinite(x);
  }
#endif



//
// wrappers for isinf


template<typename eT>
coot_inline
bool
coot_isinf(const eT&)
  {
  return false;
  }



template<>
coot_inline
bool
coot_isinf(const float& x)
  {
  return std::isinf(x);
  }



template<>
coot_inline
bool
coot_isinf(const double& x)
  {
  return std::isinf(x);
  }



template<typename T>
coot_inline
bool
coot_isinf(const std::complex<T>& x)
  {
  return ( coot_isinf(x.real()) || coot_isinf(x.imag()) );
  }



template<>
coot_inline
bool
coot_isinf(const fp16_shim& x)
  {
  return std::isinf(x.x);
  }



#if defined(COOT_USE_CUDA)
template<>
coot_inline
bool
coot_isinf(const __half& x)
  {
  #if CUDA_VERSION >= 12020
  return __hisinf(x);
  #else
  return coot_cuda_half_isinf(x);
  #endif
  }
#endif



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<>
coot_inline
bool
coot_isinf(const std::float16_t& x)
  {
  return std::isinf(x);
  }
#endif



//
// wrappers for isnan


template<typename eT>
coot_inline
bool
coot_isnan(const eT&)
  {
  return false;
  }



template<>
coot_inline
bool
coot_isnan(const float& x)
  {
  return std::isnan(x);
  }



template<>
coot_inline
bool
coot_isnan(const double& x)
  {
  return std::isnan(x);
  }



template<typename T>
coot_inline
bool
coot_isnan(const std::complex<T>& x)
  {
  return ( coot_isnan(x.real()) || coot_isnan(x.imag()) );
  }



template<>
coot_inline
bool
coot_isnan(const fp16_shim& x)
  {
  return std::isnan(x.x);
  }



#if defined(COOT_USE_CUDA)
template<>
coot_inline
bool
coot_isnan(const __half& x)
  {
  #if CUDA_VERSION >= 12020
  return __hisnan(x);
  #else
  return coot_cuda_half_isnan(x);
  #endif
  }
#endif



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<>
coot_inline
bool
coot_isnan(const std::float16_t& x)
  {
  return std::isnan(x);
  }
#endif



//
// wrappers for pow



template<typename eT>
coot_inline
eT
coot_pow(const eT& x, const eT& pow)
  {
  return std::pow(x, pow);
  }



template<>
coot_inline
fp16_shim
coot_pow(const fp16_shim& x, const fp16_shim& pow)
  {
  return fp16_shim(std::pow(x.x, pow.x));
  }



#if defined(COOT_USE_CUDA)
template<>
coot_inline
__half
coot_pow(const __half& x, const __half& p)
  {
  return __half(std::pow(float(x), float(p))); // no host __hpow() available in CUDA
  }
#endif



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<>
coot_inline
std::float16_t
coot_pow(const std::float16_t& x, const std::float16_t& p)
  {
  return std::pow(x, p);
  }
#endif



//
// wrappers for sqrt



template<typename eT>
coot_inline
eT
coot_sqrt(const eT& x)
  {
  return std::sqrt(x);
  }



template<>
coot_inline
fp16_shim
coot_sqrt(const fp16_shim& x)
  {
  return fp16_shim(std::sqrt(x.x));
  }



#if defined(COOT_USE_CUDA)
template<>
coot_inline
__half
coot_sqrt(const __half& x)
  {
  return __half(std::sqrt(float(x))); // no host __hsqrt() available in CUDA
  }
#endif



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<>
coot_inline
std::float16_t
coot_sqrt(const std::float16_t& x)
  {
  return std::sqrt(x);
  }
#endif
