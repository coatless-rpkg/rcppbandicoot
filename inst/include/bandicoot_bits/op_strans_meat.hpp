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



template<typename out_eT, typename T1>
inline
void
op_strans::apply(Mat<out_eT>& out, const Op<T1, op_strans>& in)
  {
  coot_debug_sigprint();

  // We can just directly use a proxy copy, although we need to avoid aliasing.
  Proxy<Op<T1, op_strans>> P_in(in);
  alias_wrapper<Mat<out_eT>, Proxy<Op<T1, op_strans>>> A(out, P_in);
  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());

  coot_rt_t::copy(make_proxy(A.use), make_proxy_mat(P_in));
  }



template<typename eT>
inline
void
op_strans::apply(Mat<eT>& out, const Op<Mat<eT>, op_strans>& in, const typename coot_real_only<eT>::type* junk1, const typename coot_blas_type_only<eT>::type* junk2)
  {
  coot_debug_sigprint();
  coot_ignore(junk1);
  coot_ignore(junk2);

  alias_wrapper<Mat<eT>, Mat<eT>> A(out, in.m);

  // Special case: if the output is the same as the input, we can just reset the size and no copy is needed.
  if (A.using_aux && (in.m.n_rows == 1 || in.m.n_cols == 1))
    {
    out.set_size(out.n_cols, out.n_rows);
    return;
    }

  A.use.set_size(in.m.n_cols, in.m.n_rows);
  if (get_rt().backend == CUDA_BACKEND)
    {
    // In this case, we can call the optimized cuBLAS routines for transpose.
    // For a more general T1, though, it makes more sense to unwrap it into a larger kernel.
    coot_rt_t::trans<false>(A.use.get_dev_mem(false), in.m.get_dev_mem(false), in.m.n_rows, in.m.n_cols);
    }
  else
    {
    // Use the proxy copy implementation.
    coot_rt_t::copy(make_proxy(A.use), make_proxy(in));
    }
  }



template<typename T1>
inline
uword
op_strans::compute_n_rows(const Op<T1, op_strans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }



template<typename T1>
inline
uword
op_strans::compute_n_cols(const Op<T1, op_strans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }
