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



// Utility to convert Bandicoot C++ types to OpenCL API types.

template<typename eT>
struct cl_type
  {
  typedef eT type;
  };



// The only different type for the OpenCL API is the half-precision type, where we
// use coot::fp16, which may not be the same.  However, if OpenCL isn't included,
// cl_half may not be an identifier.
#if defined(COOT_USE_OPENCL)
template<>
struct cl_type<fp16>
  {
  typedef cl_half type;
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
to_cl_type(const eT& x)
  {
  return x;
  }



// If eT is std::float16_t, then conversion to cl_half won't work correctly---we
// need to just reinterpret the value.
#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<typename eT>
inline
typename
enable_if2
  <
  is_fp16<eT>::value,
  const typename cl_type<eT>::type&
  >::result
to_cl_type(const eT& x)
  {
  return reinterpret_cast<const typename cl_type<eT>::type&>(x);
  }
#else
template<typename eT>
inline
typename
enable_if2
  <
  is_fp16<eT>::value,
  typename cl_type<eT>::type
  >::result
to_cl_type(const eT& x)
  {
  return typename cl_type<eT>::type(x);
  }
#endif



#if defined(COOT_USE_OPENCL)
template<typename out_eT, typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, cl_half>::no || is_same_type<out_eT, fp16>::no,
  const eT&
  >::result
from_cl_type(const eT& x)
  {
  return x;
  }



#if defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
template<typename out_eT, typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, cl_half>::yes && is_same_type<fp16, std::float16_t>::yes && is_same_type<out_eT, fp16>::yes,
  const fp16&
  >::result
from_cl_type(const eT& x)
  {
  return reinterpret_cast<const fp16&>(x);
  }



template<typename out_eT, typename eT>
inline
typename
enable_if2
<
  is_same_type<eT, cl_half>::yes && is_same_type<fp16, std::float16_t>::no && is_same_type<out_eT, fp16>::yes,
  fp16
  >::result
from_cl_type(const eT& x)
  {
  return fp16(x);
  }
#else
template<typename out_eT, typename eT>
inline
typename
enable_if2
  <
  is_same_type<eT, cl_half>::yes && is_same_type<out_eT, fp16>::yes,
  fp16
  >::result
from_cl_type(const eT& x)
  {
  return fp16(x);
  }
#endif



#else // defined(COOT_USE_OPENCL)



template<typename out_eT, typename eT>
inline
const eT&
from_cl_type(const eT& x)
  {
  return x;
  }



#endif
