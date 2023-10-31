// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
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



//
// matrices



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  // Our GPU kernels allow for incorporating conversion operations into eOp kernels.
  // Unfortunately, that makes setting up for these kernels a good bit more complicated!
  // But the runtime gains can be quite large by combining these operations.

  // We have five cases to handle:

  // 1. There is a type conversion after the eOp (i.e. mtOp<eT, eOp<T1, eop_type>, mtop_conv_to>).
  //    This will mean that eT != T1::elem_type.  We can handle this by passing x.aux as the 'input'
  //    aux parameter.

  // 2. There is a type conversion before the eOp (i.e. eOp<T1 = mtOp<eT, T2, mtop_conv_to>, eop_type>).
  //    This can be detected if no_conv_unwrap<T1>::stored_type::elem_type is *not* eT, but T1::elem_type is eT.
  //    We can handle this by passing x.aux as the 'output' aux parameter.

  // 3. There are type conversions before and after the eOp.  This can be detected if eT != T1::elem_type,
  //    *and* no_conv_unwrap<T1>::stored_type::elem_type != T1::elem_type.  Our kernels don't support this situation,
  //    and so we just force an additional type conversion after first handling the inner operation.

  // 4. The eOp is doubly applied, both before and after a type conversion.  (Consider input expressions like
  //    (conv_to<fmat>::from(X + 3.0d) + 2.0f), for instance.)  This can be detected via a template
  //    specialization---see the next function.  In that situation we can use the aux parameters of both the
  //    eOps as both the 'input' and 'output' aux parameters.

  // 5. There is no type conversion.  (eT == T1::elem_type, and no_conv_unwrap<T1>::stored_type::elem_type == eT.)

  // Convenience typedef.
  typedef typename no_conv_unwrap<T1>::stored_type::elem_type in_eT;

  typedef get_default<eop_type> get_default_type;

  in_eT aux_in  = get_default_type::template val<in_eT>();
     eT aux_out = get_default_type::template val<   eT>();
  bool force_conv_unwrap = false; // This will only be true if we have case 3.

  // Whether to do the conversion before or after the operation.  This only applies for a few kernel types.
  bool conv_after_op = false;

  if (is_same_type<eT, typename T1::elem_type>::no)
    {
    // This is case 1 or 3.  In both cases we set aux_in to x.aux, but if it is case 3 we must also set force_conv_unwrap.
    aux_in = x.aux;
    conv_after_op = true;

    if (is_same_type<in_eT, typename T1::elem_type>::no)
      {
      // This is case 3.
      force_conv_unwrap = true;
      aux_in = x.aux;
      }
    }
  else
    {
    // This is case 2 or 5.  In both cases, we set `aux_out` to x.aux.
    aux_out = x.aux;
    }

  const twoway_kernel_id::enum_id kernel_num = conv_after_op ? eop_type::kernel_conv_post : eop_type::kernel_conv_pre;

  dev_mem_t<eT>    out_dev_mem = out.get_dev_mem(false);

  if (!force_conv_unwrap)
    {
    const no_conv_unwrap<typename SizeProxy<T1>::stored_type> U(x.m.Q);

    dev_mem_t<in_eT> A_dev_mem = U.get_dev_mem(false);

    coot_rt_t::eop_scalar(kernel_num, out_dev_mem, A_dev_mem,
                          aux_in, aux_out,
                          out.n_rows, out.n_cols,
                          0, 0, out.get_n_rows(),
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  else
    {
    // We have to perform any conversion before this level.
    const unwrap<typename SizeProxy<T1>::stored_type> U(x.m.Q);

    dev_mem_t<typename T1::elem_type> A_dev_mem = U.get_dev_mem(false);

    coot_rt_t::eop_scalar(kernel_num, out_dev_mem, A_dev_mem,
                          (typename T1::elem_type) aux_in, aux_out,
                          out.n_rows, out.n_cols,
                          0, 0, out.n_rows,
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  }



// This specialization is for case 4 (described above).
template<typename eop_type>
template<typename eT, typename T2>
inline
void
eop_core<eop_type>::apply(Mat<eT>& out, const eOp<mtOp<eT, eOp<T2, eop_type>, mtop_conv_to>, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T2::elem_type in_eT;

  in_eT aux_in  = X.m.Q.q.aux;
     eT aux_out = X.aux;

  // Pretend that we're doing the conversion after the operation.
  const twoway_kernel_id::enum_id kernel_num = eop_type::kernel_conv_post;

  const unwrap<T2> U(X.m.Q.q.m.Q);

  dev_mem_t<eT>    out_dev_mem = out.get_dev_mem(false);
  dev_mem_t<in_eT>   A_dev_mem = U.get_dev_mem(false);

  // There are a couple exceptions of operations where we actually *can't* chain them together (because the kernels
  // themselves can't support it).
  if (!eop_type::is_chainable)
    {
    Mat<in_eT> tmp(U.M.n_rows, U.M.n_cols);
    coot_rt_t::eop_scalar(kernel_num, tmp.get_dev_mem(), A_dev_mem,
                          aux_in, in_eT(0),
                          tmp.n_rows, tmp.n_cols,
                          0, 0, tmp.n_rows,
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    coot_rt_t::eop_scalar(kernel_num, out_dev_mem, tmp.get_dev_mem(),
                          in_eT(0), aux_out,
                          out.n_rows, out.n_cols,
                          0, 0, out.n_rows,
                          0, 0, tmp.n_rows);

    return;
    }
  else
    {
    coot_rt_t::eop_scalar(kernel_num, out_dev_mem, A_dev_mem,
                          aux_in, aux_out,
                          out.n_rows, out.n_cols,
                          0, 0, out.n_rows,
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  }



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_plus(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "addition");

  const Mat<eT> tmp(x);

  out += tmp;
  }



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_minus(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "subtraction");

  const Mat<eT> tmp(x);

  out -= tmp;
  }



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_schur(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise multiplication");

  const Mat<eT> tmp(x);

  out %= tmp;
  }



template<typename eop_type>
template<typename eT, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_div(Mat<eT>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise division");

  const Mat<eT> tmp(x);

  out /= tmp;
  }
