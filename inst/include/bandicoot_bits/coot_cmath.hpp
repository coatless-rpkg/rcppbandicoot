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
coot_isfinite(eT)
  {
  return true;
  }



template<>
coot_inline
bool
coot_isfinite(float x)
  {
  return std::isfinite(x);
  }



template<>
coot_inline
bool
coot_isfinite(double x)
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



//
// wrappers for isinf


template<typename eT>
coot_inline
bool
coot_isinf(eT)
  {
  return false;
  }



template<>
coot_inline
bool
coot_isinf(float x)
  {
  return std::isinf(x);
  }



template<>
coot_inline
bool
coot_isinf(double x)
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



//
// wrappers for isnan


template<typename eT>
coot_inline
bool
coot_isnan(eT)
  {
  return false;
  }



template<>
coot_inline
bool
coot_isnan(float x)
  {
  return std::isnan(x);
  }



template<>
coot_inline
bool
coot_isnan(double x)
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



