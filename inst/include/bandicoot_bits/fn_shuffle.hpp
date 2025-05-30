// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value && resolves_to_vector<T1>::value,
  const Op<T1, op_shuffle_vec>
  >::result
shuffle
  (
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_shuffle_vec>(X);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value && !resolves_to_vector<T1>::value,
  const Op<T1, op_shuffle>
  >::result
shuffle
  (
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_shuffle>(X, 0, 0);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  (is_coot_type<T1>::value),
  const Op<T1, op_shuffle>
  >::result
shuffle
  (
  const T1&   X,
  const uword dim
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_shuffle>(X, dim, 0);
  }
