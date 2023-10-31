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
const mtOp<uword, T1, mtop_sort_index>
sort_index
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_sort_index>(X.get_ref(), 0, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  ( (is_coot_type<T1>::value) && (is_same_type<T2, char>::value) ),
  const mtOp<uword, T1, mtop_sort_index>
  >::result
sort_index
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (sort_direction != nullptr) ? sort_direction[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'd')), "sort_index(): unknown sort direction" );

  return mtOp<uword, T1, mtop_sort_index>(X, ((sig == 'a') ? 0 : 1), 0);
  }



template<typename T1>
coot_warn_unused
inline
const mtOp<uword, T1, mtop_sort_index>
stable_sort_index
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_sort_index>(X.get_ref(), 0, 1);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  ( (is_coot_type<T1>::value) && (is_same_type<T2, char>::value) ),
  const mtOp<uword, T1, mtop_sort_index>
  >::result
stable_sort_index
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (sort_direction != nullptr) ? sort_direction[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'd')), "stable_sort_index(): unknown sort direction" );

  return mtOp<uword, T1, mtop_sort_index>(X, ((sig == 'a') ? 0 : 1), 1);
  }
