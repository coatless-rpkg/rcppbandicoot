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
inline
void
op_stddev::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_stddev>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const uword norm_type = in.aux_uword_a;
  const uword dim = in.aux_uword_b;

  // The kernels we have don't operate on subviews, or aliases.
  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);
  const copy_alias<eT> C(E.M, out);

  // First compute the variance.
  op_var::apply_direct(out, C.M, dim, norm_type);

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // Now take the square root.
  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_sqrt_pre, out.get_dev_mem(false), out.get_dev_mem(false),
                        eT(0), eT(0),
                        out.n_rows, out.n_cols, 1,
                        0, 0, 0, out.n_rows, out.n_cols,
                        0, 0, 0, out.n_rows, out.n_cols);
  }



template<typename out_eT, typename T1>
inline
void
op_stddev::apply(Mat<out_eT>& out, const Op<T1, op_stddev>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  const uword norm_type = in.aux_uword_a;
  const uword dim = in.aux_uword_b;

  const unwrap<T1> U(in.m);

  // If there is a type conversion, we must first compute using the original element type, and then convert in the last step.
  Mat<eT> tmp;
  op_var::apply_direct(tmp, U.M, dim, norm_type);
  out.set_size(tmp.n_rows, tmp.n_cols);

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // Now take the square root.
  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_sqrt_post, out.get_dev_mem(false), tmp.get_dev_mem(false),
                        eT(0), out_eT(0),
                        out.n_rows, out.n_cols, 1,
                        0, 0, 0, out.n_rows, out.n_cols,
                        0, 0, 0, tmp.n_rows, tmp.n_cols);
  }



template<typename T1>
inline
uword
op_stddev::compute_n_rows(const Op<T1, op_stddev>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return std::min(in_n_rows, uword(1)); // either 0 or 1
    }
  else
    {
    return in_n_rows;
    }
  }



template<typename T1>
inline
uword
op_stddev::compute_n_cols(const Op<T1, op_stddev>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return in_n_cols;
    }
  else
    {
    return std::min(in_n_cols, uword(1)); // either 0 or 1
    }
  }
