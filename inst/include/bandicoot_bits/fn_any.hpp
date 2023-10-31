// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
// Copyright 2017 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename T1>
coot_warn_unused
inline
typename enable_if2<
  is_coot_type<T1>::value && resolves_to_vector<T1>::value,
  bool
>::result
any(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtop_any::any_vec(X);
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2<
  is_coot_type<T1>::value && !resolves_to_vector<T1>::value,
  const mtOp<uword, T1, mtop_any>
>::result
any(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_any>(X, 0, 0);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2<
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_any>
>::result
any(const T1& X, const uword dim)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_any>(X, dim, 0);
  }


// for compatibility purposes: allows compiling user code designed for earlier versions of Armadillo
template<typename T>
coot_warn_unused
inline
typename
enable_if2
  <
  is_supported_elem_type<T>::value,
  bool
  >::result
any(const T& val)
  {
  return (val != T(0));
  }
