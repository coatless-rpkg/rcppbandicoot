// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022-2025 Ryan Curtin (http://www.ratml.org/)
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



template<typename out_eT, typename T1>
inline
void
op_repmat::apply(Mat<out_eT>& out, const Op<T1, op_repmat>& in)
  {
  coot_debug_sigprint();

  Proxy<Op<T1, op_repmat>> P_in(in);

  alias_wrapper<Mat<out_eT>, Proxy<Op<T1, op_repmat>> > W(out, P_in);

  // Skip if there is nothing to do.
  if (W.using_aux && in.aux_uword_a == 1 && in.aux_uword_b == 1 && is_same_type<out_eT, typename T1::elem_type>::yes)
    {
    W.using_aux = false; // disable steal_mem() in destructor
    return;
    }

  W.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(W.use), P_in);
  }



template<typename T1>
inline
uword
op_repmat::compute_n_rows(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);
  return op.aux_uword_a * in_n_rows;
  }



template<typename T1>
inline
uword
op_repmat::compute_n_cols(const Op<T1, op_repmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);
  return op.aux_uword_b * in_n_cols;
  }
