// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://www.ratml.org)
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



template<typename T1, typename op_type>
inline
CubeToMatOp<T1, op_type>::CubeToMatOp(const T1& in_m)
  : m(in_m)
  {
  coot_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
inline
CubeToMatOp<T1, op_type>::CubeToMatOp(const T1& in_m, const uword in_aux_uword)
  : m(in_m)
  , aux_uword(in_aux_uword)
  {
  coot_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
inline
CubeToMatOp<T1, op_type>::~CubeToMatOp()
  {
  coot_extra_debug_sigprint();
  }
