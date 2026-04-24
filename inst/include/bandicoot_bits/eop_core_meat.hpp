// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2026 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_debug_sigprint();

  const Proxy<eOp<T1, eop_type>> P_in(x);
  const inexact_alias_wrapper<Mat<eT>, Proxy<eOp<T1, eop_type>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply(Cube<eT>& out, const eOpCube<T1, eop_type>& x)
  {
  coot_debug_sigprint();

  const Proxy<eOpCube<T1, eop_type>> P_in(x);
  const inexact_alias_wrapper<Cube<eT>, Proxy<eOpCube<T1, eop_type>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols(), P_in.get_n_slices());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }
