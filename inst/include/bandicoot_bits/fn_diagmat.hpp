// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023      Ryan Curtin (http://www.ratml.org)
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



//! interpret a matrix or a vector as a diagonal matrix (ie. off-diagonal entries are zero)
template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
diagmat(const T1& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_diagmat>(X);
  }



//! create a matrix with the k-th diagonal set to the given vector
template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat2>
  >::result
diagmat(const T1& X, const sword k)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 1 : 0;

  return Op<T1, op_diagmat2>(X, a, b);
  }



// simplification: wrap transposes into diagmat
template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
diagmat(const Op<T1, op_htrans>& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_diagmat>(X.m);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat2>
  >::result
diagmat(const Op<T1, op_htrans>& X, const sword k)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 3 : 2;

  return Op<T1, op_diagmat2>(X.m, a, b);
  }
