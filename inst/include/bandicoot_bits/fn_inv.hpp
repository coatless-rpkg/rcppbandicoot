// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023-2026 Ryan Curtin (http://www.ratml.org)
// Copyright 2023-2026 Conrad Sanderson (https://conradsanderson.id.au)
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
enable_if2<
  is_real<typename T1::elem_type>::value,
  const Op<T1, op_inv_gen_default>
>::result
inv
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_debug_sigprint();
  
  return Op<T1, op_inv_gen_default>(X.get_ref());
  }



template<typename T1>
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  bool
>::result
inv
  (
         Mat<typename T1::elem_type>&     out,
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const std::tuple<bool, std::string> result = op_inv_gen_default::apply_direct(out, X.get_ref());
  
  if(std::get<0>(result) == false)
    {
    out.reset();
    coot_warn(3, "inv(): " + std::get<1>(result));
    }
  
  return std::get<0>(result);
  }
