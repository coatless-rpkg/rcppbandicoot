// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2025 Ryan Curtin (https://www.ratml.org)
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



// unary plus operation (does nothing, but is required for completeness)
template<typename T1>
coot_inline
const BaseCube<typename T1::elem_type,T1>&
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return X;
  }



// BaseCube + scalar
template<typename T1>
coot_inline
const eOpCube<T1, eop_scalar_plus>
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const typename T1::elem_type               k
  )
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_scalar_plus>(X.get_ref(), k);
  }



// scalar + BaseCube
template<typename T1>
coot_inline
const eOpCube<T1, eop_scalar_plus>
operator+
  (
  const typename T1::elem_type               k,
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return eOpCube<T1, eop_scalar_plus>(X.get_ref(), k);
  }



// addition of BaseCube objects with same element type
template<typename T1, typename T2>
coot_inline
const eGlueCube<T1, T2, eglue_plus>
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const BaseCube<typename T1::elem_type,T2>& Y
  )
  {
  coot_extra_debug_sigprint();

  return eGlueCube<T1, T2, eglue_plus>(X.get_ref(), Y.get_ref());
  }


// addition of Base objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_coot_cube_type<T1>::value && is_coot_cube_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const eGlueCube<T1, T2, glue_mixed_plus>
  >::result
operator+
  (
  const T1& X,
  const T2& Y
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;

  promote_type<eT1,eT2>::check();

  return eGlueCube<T1, T2, glue_mixed_plus>( X, Y );
  }
