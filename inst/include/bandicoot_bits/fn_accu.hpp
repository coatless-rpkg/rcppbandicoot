// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename eT>
coot_warn_unused
inline
eT
accu(const Mat<eT>& A)
  {
  coot_extra_debug_sigprint();

  if (A.n_elem == 0) { return eT(0); }

  return coot_rt_t::accu(A.get_dev_mem(false), A.n_elem);
  }



template<typename eT>
coot_warn_unused
inline
eT
accu(const subview<eT>& S)
  {
  coot_extra_debug_sigprint();

  if(S.n_elem == 0)  { return eT(0); }

  return coot_rt_t::accu_subview(S.m.get_dev_mem(false), S.m.n_rows, S.aux_row1, S.aux_col1, S.n_rows, S.n_cols);
  }



template<typename T1>
coot_warn_unused
inline
typename T1::elem_type
accu(const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.get_ref());
  return accu(U.M);
  }
