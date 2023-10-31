// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023      Ryan Curtin (http://www.ratml.org/)
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



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value && (resolves_to_vector<T1>::value == true),
  const Op<T1, op_sort_vec>
  >::result
sort
  (
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_sort_vec>(X, 0, 0);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value && (resolves_to_vector<T1>::value == false),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_sort>(X, 0, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value && (resolves_to_vector<T1>::value == true) && is_same_type<T2, char>::value,
  const Op<T1, op_sort_vec>
  >::result
sort
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (sort_direction != nullptr) ? sort_direction[0] : char(0);

  coot_debug_check( (sig != 'a') && (sig != 'd'), "sort(): unknown sort direction" );

  const uword sort_type = (sig == 'a') ? 0 : 1;

  return Op<T1, op_sort_vec>(X, sort_type, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value && (resolves_to_vector<T1>::value == false) && is_same_type<T2, char>::value,
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (sort_direction != nullptr) ? sort_direction[0] : char(0);

  coot_debug_check( (sig != 'a') && (sig != 'd'), "sort(): unknown sort direction" );

  const uword sort_type = (sig == 'a') ? 0 : 1;

  return Op<T1, op_sort>(X, sort_type, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  ( (is_coot_type<T1>::value) && (is_same_type<T2, char>::value) ),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1&   X,
  const T2*   sort_direction,
  const uword dim
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (sort_direction != nullptr) ? sort_direction[0] : char(0);

  coot_debug_check( (sig != 'a') && (sig != 'd'), "sort(): unknown sort direction" );

  const uword sort_type = (sig == 'a') ? 0 : 1;

  return Op<T1, op_sort>(X, sort_type, dim);
  }
