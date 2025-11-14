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



// Utility to convert Bandicoot C++ types to CUDA API types.

template<typename eT>
struct cuda_type
  {
  typedef eT type;
  };



// The only different type for the CUDA API is the half-precision type, where we
// use coot::fp16, which may not be the same.  However, if CUDA isn't included,
// __half may not be an identifier.
#if defined(COOT_USE_CUDA)
template<>
struct cuda_type<fp16>
  {
  typedef __half type;
  };
#endif



template<typename eT>
inline
typename
enable_if2
  <
  !is_fp16<eT>::value,
  const eT&
  >::result
to_cuda_type(const eT& x)
  {
  return x;
  }



// If eT is std::float16_t, then CUDA doesn't have a conversion operator.
// However, we can just reinterpret the existing version.a
#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<typename eT>
inline
typename
enable_if2
  <
  is_fp16<eT>::value,
  const typename cuda_type<eT>::type&
  >::result
to_cuda_type(const eT& x)
  {
  return reinterpret_cast<const typename cuda_type<eT>::type&>(x);
  }
#else
template<typename eT>
inline
typename
enable_if2
  <
  is_fp16<eT>::value,
  typename cuda_type<eT>::type
  >::result
to_cuda_type(const eT& x)
  {
  return typename cuda_type<eT>::type(x);
  }
#endif



#if defined(COOT_USE_CUDA)
template<typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, __half>::no,
  const eT&
  >::result
from_cuda_type(const eT& x)
  {
  return x;
  }



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, __half>::yes && is_same_type<fp16, std::float16_t>::yes,
  const fp16&
  >::result
from_cuda_type(const eT& x)
  {
  return reinterpret_cast<const fp16&>(x);
  }



template<typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, __half>::yes && is_same_type<fp16, std::float16_t>::no,
  fp16
  >::result
from_cuda_type(const eT& x)
  {
  return fp16(x);
  }
#else
template<typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, __half>::yes,
  fp16
  >::result
from_cuda_type(const eT& x)
  {
  return fp16(x);
  }
#endif



#else // defined(COOT_USE_CUDA)



template<typename eT>
inline
const eT&
from_cuda_type(const eT& x)
  {
  return x;
  }



#endif
