// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023      Ryan Curtin (https://www.ratml.org)
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
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_find>
  >::result
find(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_find>(X);
  }



template<typename T1>
coot_warn_unused
inline
const mtOp<uword, T1, mtop_find>
find
  (
  const Base<typename T1::elem_type, T1>& X,
  const uword k,
  const char* direction = "first"
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (direction != nullptr) ? direction[0] : char(0);

  coot_debug_check
    (
    ( (sig != 'f') && (sig != 'F') && (sig != 'l') && (sig != 'L') ),
    "find(): direction must be \"first\" or \"last\""
    );

  const uword type = ( (sig == 'f') || (sig == 'F') ) ? 0 : 1;

  return mtOp<uword, T1, mtop_find>(X.get_ref(), k, type);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_find_finite>
  >::result
find_finite(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_find_finite>(X);
  }



template<typename T1>
coot_warn_unused
inline
const mtOp<uword, T1, mtop_find_finite>
find_finite
  (
  const Base<typename T1::elem_type, T1>& X,
  const uword k,
  const char* direction = "first"
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (direction != nullptr) ? direction[0] : char(0);

  coot_debug_check
    (
    ( (sig != 'f') && (sig != 'F') && (sig != 'l') && (sig != 'L') ),
    "find_finite(): direction must be \"first\" or \"last\""
    );

  const uword type = ( (sig == 'f') || (sig == 'F') ) ? 0 : 1;

  return mtOp<uword, T1, mtop_find_finite>(X.get_ref(), k, type);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_find_nonfinite>
  >::result
find_nonfinite(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_find_nonfinite>(X);
  }



template<typename T1>
coot_warn_unused
inline
const mtOp<uword, T1, mtop_find_nonfinite>
find_nonfinite
  (
  const Base<typename T1::elem_type, T1>& X,
  const uword k,
  const char* direction = "first"
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (direction != nullptr) ? direction[0] : char(0);

  coot_debug_check
    (
    ( (sig != 'f') && (sig != 'F') && (sig != 'l') && (sig != 'L') ),
    "find_nonfinite(): direction must be \"first\" or \"last\""
    );

  const uword type = ( (sig == 'f') || (sig == 'F') ) ? 0 : 1;

  return mtOp<uword, T1, mtop_find_nonfinite>(X.get_ref(), k, type);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_find_nan>
  >::result
find_nan(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_find_nan>(X);
  }



template<typename T1>
coot_warn_unused
inline
const mtOp<uword, T1, mtop_find_nan>
find_nan
  (
  const Base<typename T1::elem_type, T1>& X,
  const uword k,
  const char* direction = "first"
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (direction != nullptr) ? direction[0] : char(0);

  coot_debug_check
    (
    ( (sig != 'f') && (sig != 'F') && (sig != 'l') && (sig != 'L') ),
    "find_nan(): direction must be \"first\" or \"last\""
    );

  const uword type = ( (sig == 'f') || (sig == 'F') ) ? 0 : 1;

  return mtOp<uword, T1, mtop_find_nan>(X.get_ref(), k, type);
  }



