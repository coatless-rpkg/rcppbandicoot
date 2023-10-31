// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
const Op<T1, op_symmat>
symmatu(const Base<typename T1::elem_type, T1>& X, const bool do_conj = true)
  {
  coot_extra_debug_sigprint();
  coot_ignore(do_conj);

  return Op<T1, op_symmat>(X.get_ref(), 0, 0);
  }



template<typename T1>
coot_warn_unused
inline
const Op<T1, op_symmat>
symmatl(const Base<typename T1::elem_type, T1>& X, const bool do_conj = true)
  {
  coot_extra_debug_sigprint();
  coot_ignore(do_conj);

  return Op<T1, op_symmat>(X.get_ref(), 1, 0);
  }
