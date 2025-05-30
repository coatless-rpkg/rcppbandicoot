// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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
typename enable_if2< is_coot_type<T1>::value && resolves_to_vector<T1>::value == true, uword>::result
index_min(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtop_index_min::apply_direct(X);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value && resolves_to_vector<T1>::value == false, const mtOp<uword, T1, mtop_index_min> >::result
index_min(const T1& X)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_index_min>(X, 0, 0);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const mtOp<uword, T1, mtop_index_min> >::result
index_min(const T1& X, const uword dim)
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_index_min>(X, dim, 0);
  }



coot_warn_unused
inline
uword
index_min(const SizeMat& s)
  {
  return (s.n_rows <= s.n_cols) ? uword(0) : uword(1);
  }



coot_warn_unused
inline
uword
index_min(const SizeCube& s)
  {
  const uword tmp_val   = (s.n_rows <= s.n_cols) ? s.n_rows : s.n_cols;
  const uword tmp_index = (s.n_rows <= s.n_cols) ? uword(0) : uword(1);

  return (tmp_val <= s.n_slices) ? tmp_index : uword(2);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_cube_type<T1>::value, const mtOpCube<uword, T1, mtop_index_min> >::result
index_min(const T1& X, const uword dim)
  {
  coot_extra_debug_sigprint();

  return mtOpCube<uword, T1, mtop_index_min>(X, dim, 0);
  }
