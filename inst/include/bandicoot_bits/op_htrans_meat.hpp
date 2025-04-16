// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
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
op_htrans::apply(Mat<out_eT>& out, const Op<T1, op_htrans>& in)
  {
  coot_extra_debug_sigprint();

  const no_conv_unwrap<T1> U(in.m);
  const extract_subview<typename no_conv_unwrap<T1>::stored_type> E(U.M);

  alias_wrapper<Mat<out_eT>, Mat<typename no_conv_unwrap<T1>::stored_type::elem_type>> W(out, E.M);
  W.use.set_size(E.M.n_cols, E.M.n_rows);
  if (W.use.n_elem == 0)
    {
    return;
    }

  if (E.M.n_cols == 1 || E.M.n_rows == 1)
    {
    // Simply copying the data is sufficient.
    coot_rt_t::copy_mat(W.get_dev_mem(false), E.M.get_dev_mem(false),
                        // logically treat both as vectors
                        W.use.n_elem, 1,
                        0, 0, W.use.n_elem,
                        0, 0, E.M.n_elem);
    }
  else
    {
    coot_rt_t::htrans(W.use.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_rows, E.M.n_cols);
    }
  }



template<typename T1>
inline
uword
op_htrans::compute_n_rows(const Op<T1, op_htrans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }



template<typename T1>
inline
uword
op_htrans::compute_n_cols(const Op<T1, op_htrans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }



//



template<typename out_eT, typename T1>
inline
void
op_htrans2::apply(Mat<out_eT>& out, const Op<T1, op_htrans2>& in)
  {
  coot_extra_debug_sigprint();

  op_htrans::apply(out, Op<T1, op_htrans>(in.m));
  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                        out.get_dev_mem(false), out.get_dev_mem(false),
                        in.aux, (out_eT) 1,
                        out.n_rows, out.n_cols, 1,
                        0, 0, 0, out.n_rows, out.n_cols,
                        0, 0, 0, out.n_rows, out.n_cols);
  }



template<typename T1>
inline
uword
op_htrans2::compute_n_rows(const Op<T1, op_htrans2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }



template<typename T1>
inline
uword
op_htrans2::compute_n_cols(const Op<T1, op_htrans2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }
