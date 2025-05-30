// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2021-2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename T1>
coot_warn_unused
inline
const Op<T1, op_min>
min
  (
  const T1& X,
  const uword dim = 0,
  const typename enable_if< is_coot_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk1);
  coot_ignore(junk2);

  return Op<T1, op_min>(X, dim, 0);
  }



template<typename T1>
coot_warn_unused
inline
const Op<T1, op_min>
min
  (
  const T1& X,
  const uword dim,
  const typename enable_if< resolves_to_vector<T1>::value == true >::result* junk = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return Op<T1, op_min>(X, dim, 0);
  }



template<typename T1>
coot_warn_unused
inline
typename T1::elem_type
min
  (
  const T1& X,
  const coot_empty_class junk1 = coot_empty_class(),
  const typename enable_if< resolves_to_vector<T1>::value == true >::result* junk2 = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk1);
  coot_ignore(junk2);

  return op_min::apply_direct(X);
  }



template<typename T1>
coot_warn_unused
inline
typename T1::elem_type
min
  (
  const Op<T1, op_min>& in
  )
  {
  coot_extra_debug_sigprint();
  coot_extra_debug_print("min(): two consecutive min() calls detected");

  return op_min::apply_direct(in.m);
  }



coot_warn_unused
inline
uword
min(const SizeMat& s)
  {
  return (std::min)(s.n_rows, s.n_cols);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  ( is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value ),
  Glue<T1, T2, glue_min>
  >::result
min
  (
  const T1& X,
  const T2& Y
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_min>(X, Y);
  }



coot_warn_unused
inline
uword
min(const SizeCube& s)
  {
  return (std::min)( (std::min)(s.n_rows, s.n_cols), s.n_slices );
  }



template<typename T1>
coot_warn_unused
inline
const OpCube<T1, op_min>
min
  (
  const T1& X,
  const uword dim = 0,
  const typename enable_if< is_coot_cube_type<T1>::value  == true  >::result* junk = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return OpCube<T1, op_min>(X, dim, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  ( is_coot_cube_type<T1>::value && is_coot_cube_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value ),
  GlueCube<T1, T2, glue_min>
  >::result
min
  (
  const T1& X,
  const T2& Y
  )
  {
  coot_extra_debug_sigprint();

  return GlueCube<T1, T2, glue_min>(X, Y);
  }
