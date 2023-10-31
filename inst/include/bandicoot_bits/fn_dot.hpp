// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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



template<typename eT1, typename eT2, typename T1, typename T2>
coot_warn_unused
inline
typename promote_type<eT1, eT2>::result
dot
  (
  const Base<eT1, T1>& A,
  const Base<eT2, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  const unwrap<T1>    U(A.get_ref());
  const unwrap<T2>    V(B.get_ref());

  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  const extract_subview<typename unwrap<T2>::stored_type> F(V.M);

  const Mat<eT1>& X = E.M;
  const Mat<eT2>& Y = F.M;

  // check same size

  return coot_rt_t::dot(X.get_dev_mem(false), Y.get_dev_mem(false), X.n_elem);
  }
