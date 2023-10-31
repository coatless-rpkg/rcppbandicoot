// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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
const Op<T1, op_htrans>
trans(const Base<typename T1::elem_type,T1>& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_htrans>(X.get_ref());
  }



template<typename T1>
coot_warn_unused
inline
const Op<T1, op_htrans>
htrans(const Base<typename T1::elem_type,T1>& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_htrans>(X.get_ref());
  }



// simplification: trans(diagmat()) does nothing

template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
trans(const Op<T1, op_diagmat>& X)
  {
  coot_extra_debug_sigprint();

  return X;
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
htrans(const Op<T1, op_diagmat>& X)
  {
  coot_extra_debug_sigprint();

  return X;
  }
