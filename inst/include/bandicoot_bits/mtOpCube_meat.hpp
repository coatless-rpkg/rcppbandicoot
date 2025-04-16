// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename out_eT, typename T1, typename mtop_type>
inline
mtOpCube<out_eT, T1, mtop_type>::mtOpCube(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : q(in_m)
  , aux(out_eT(0))
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename mtop_type>
inline
mtOpCube<out_eT, T1, mtop_type>::mtOpCube(const T1& in_m, const typename T1::elem_type aux_in)
  : q(in_m)
  , aux(aux_in)
  , aux_uword_a(0)
  , aux_uword_b(0)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename mtop_type>
inline
mtOpCube<out_eT, T1, mtop_type>::~mtOpCube()
  {
  coot_extra_debug_sigprint();
  }
