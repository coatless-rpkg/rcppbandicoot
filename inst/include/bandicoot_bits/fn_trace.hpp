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
typename T1::elem_type
trace(const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap<T1>   U(X.get_ref());
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  const Mat<eT>& A = E.M;

  if(A.n_elem == 0)  { return eT(0); }

  return coot_rt_t::trace(A.get_dev_mem(false), A.n_rows, A.n_cols);
  }



// trace(diagmat): just sum the elements
template<typename T1>
coot_warn_unused
inline
typename T1::elem_type
trace(const Op<T1, op_diagmat>& X)
  {
  coot_extra_debug_sigprint();

  return accu(X.m);
  }
