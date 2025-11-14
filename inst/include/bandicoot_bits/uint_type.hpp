// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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


// Utilities to get an unsigned integer type of the same width as the given type.




template<typename T, bool integral> struct uint_type_helper { };

template<typename T> struct uint_type_helper<T, true> { typedef typename std::make_unsigned<T>::type result; };

template<> struct uint_type_helper<fp16,                 false> { typedef u16  result; };
template<> struct uint_type_helper<float,                false> { typedef u32  result; };
template<> struct uint_type_helper<double,               false> { typedef u64  result; };
template<> struct uint_type_helper<std::complex<float>,  false> { typedef u32  result; };
template<> struct uint_type_helper<std::complex<double>, false> { typedef u64  result; };
// Used sometimes by the kernel generation utilities to avoid specifying an unnecessary type.
template<> struct uint_type_helper<void,   false> { typedef void result; };



template<typename T> struct uint_type
  {
  typedef typename uint_type_helper<T, !(is_real<T>::value || is_cx<T>::value)>::result result;
  };
