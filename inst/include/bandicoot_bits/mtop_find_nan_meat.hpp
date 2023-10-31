// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
mtop_find_nan::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_find_nan>& in)
  {
  coot_extra_debug_sigprint();

  if (in.is_computed)
    {
    out.steal_mem(const_cast<Mat<uword>&>(in.computed_result));
    return;
    }

  typedef typename T1::elem_type eT;

  if (!is_real<eT>::value)
    {
    // If the type is not a real type, then all elements are not NaNs.
    out.reset();
    return;
    }

  unwrap<T1> U(in.q);
  extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  // For the first step, we have to find all finite values.
  Mat<uword> X(E.M.n_rows, E.M.n_cols);
  coot_rt_t::relational_unary_array_op(X.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_elem, oneway_real_kernel_id::rel_isnan, "find_nan");

  const uword k         = in.aux_uword_a;
  const uword find_type = in.aux_uword_b;

  out.reset(); // release any current memory

  uword result_size;
  dev_mem_t<uword> out_mem;
  coot_rt_t::find(out_mem, result_size, X.get_dev_mem(false), X.n_elem, k, find_type);

  out = Mat<uword>(out_mem, result_size, 1);
  // Take ownership of the memory that got allocated.
  access::rw(out.mem_state) = 0;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_find_nan::compute_n_rows(const mtOp<out_eT, T1, mtop_find_nan>& op, const uword in_n_rows, const uword in_n_cols)
  {
  // We can't know the size of the result unless we actually compute it, unfortunately.
  apply(const_cast<Mat<uword>&>(op.computed_result), op);
  access::rw(op.is_computed) = true;

  return op.computed_result.n_elem;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_find_nan::compute_n_cols(const mtOp<out_eT, T1, mtop_find_nan>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return 1;
  }
