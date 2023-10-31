// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_solve::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_solve>& in)
  {
  coot_extra_debug_sigprint();

  const std::tuple<bool, std::string> result = apply(out, in.A, in.B, in.aux_uword);
  if (std::get<0>(result) == false)
    {
    out.reset();
    coot_stop_runtime_error("solve(): " + std::get<1>(result));
    }
  }



template<typename out_eT, typename eT, typename T1, typename T2>
inline
std::tuple<bool, std::string>
glue_solve::apply(Mat<out_eT>& out, const Base<eT, T1>& A_expr, const Base<eT, T2>& B_expr, const uword flags, const typename enable_if<!is_same_type<eT, out_eT>::value>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  Mat<eT> tmp_out;
  const std::tuple<bool, std::string> result = apply(tmp_out, A_expr, B_expr, flags);
  if (std::get<0>(result) == true)
    {
    // convert to output type
    out.set_size(tmp_out.n_rows, tmp_out.n_cols);
    coot_rt_t::copy_mat(out.get_dev_mem(false), tmp_out.get_dev_mem(false),
                        tmp_out.n_rows, tmp_out.n_cols,
                        0, 0, out.n_rows,
                        0, 0, tmp_out.n_rows);
    }

  return result;
  }



template<typename eT, typename T1, typename T2>
inline
std::tuple<bool, std::string>
glue_solve::apply(Mat<eT>& out, const Base<eT, T1>& A_expr, const Base<eT, T2>& B_expr, const uword flags)
  {
  coot_extra_debug_sigprint();

  coot_ignore(flags);  // TODO: right now we only support LU-based decompositions, and so flags are ignored

  Mat<eT> A(A_expr.get_ref()); // A needs to be unwrapped into a new matrix, that will be destroyed during computation
  out = B_expr.get_ref(); // form B into `out`; solve_square_fast() will overwrite with the result

  if (A.is_empty() || out.is_empty())
    {
    out.set_size(A.n_rows, out.n_cols);
    return std::make_tuple(true, "");
    }

  coot_debug_check( A.n_rows != A.n_cols, "solve(): given matrix must be square sized" );

  copy_alias<eT> A2(A, out);
  return coot_rt_t::solve_square_fast(A2.M.get_dev_mem(true), false, out.get_dev_mem(true), out.n_rows, out.n_cols);
  }



template<typename eT, typename T1, typename T2>
inline
std::tuple<bool, std::string>
glue_solve::apply(Mat<eT>& out, const Base<eT, Op<T1, op_htrans>>& A_expr, const Base<eT, T2>& B_expr, const uword flags)
  {
  coot_extra_debug_sigprint();

  coot_ignore(flags);  // TODO: right now we only support LU-based decompositions, and so flags are ignored

  Mat<eT> A(A_expr.get_ref().m); // A needs to be unwrapped into a new matrix, that will be destroyed during computation
  out = B_expr.get_ref(); // form B into `out`; solve_square_fast() will overwrite with the result

  if (A.is_empty() || out.is_empty())
    {
    out.set_size(A.n_rows, out.n_cols);
    return std::make_tuple(true, "");
    }

  coot_debug_check( A.n_rows != A.n_cols, "solve(): given matrix must be square sized" );

  copy_alias<eT> A2(A, out);
  return coot_rt_t::solve_square_fast(A2.M.get_dev_mem(true), true, out.get_dev_mem(true), out.n_rows, out.n_cols);
  }



template<typename eT, typename T1, typename T2>
inline
std::tuple<bool, std::string>
glue_solve::apply(Mat<eT>& out, const Base<eT, Op<T1, op_htrans2>>& A_expr, const Base<eT, T2>& B_expr, const uword flags)
  {
  coot_extra_debug_sigprint();

  coot_ignore(flags);  // TODO: right now we only support LU-based decompositions, and so flags are ignored

  Mat<eT> A(A_expr.get_ref().m); // A needs to be unwrapped into a new matrix, that will be destroyed during computation
  out = B_expr.get_ref(); // form B into `out`; solve_square_fast() will overwrite with the result

  if (A.is_empty() || out.is_empty())
    {
    out.set_size(A.n_rows, out.n_cols);
    return std::make_tuple(true, "");
    }

  coot_debug_check( A.n_rows != A.n_cols, "solve(): given matrix must be square sized" );

  copy_alias<eT> A2(A, out);
  const std::tuple<bool, std::string> result = coot_rt_t::solve_square_fast(A2.M.get_dev_mem(true), true, out.get_dev_mem(true), out.n_rows, out.n_cols);
  if (std::get<0>(result) == true)
    {
    // the result needs to be divided by the scalar
    coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                          out.get_dev_mem(false), out.get_dev_mem(false),
                          (eT) A_expr.get_ref().aux, (eT) 1,
                          out.n_rows, out.n_cols,
                          0, 0, out.n_rows,
                          0, 0, out.n_rows);
    }

  return result;
  }



template<typename T1, typename T2>
inline
uword
glue_solve::compute_n_rows(const Glue<T1, T2, glue_solve>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_cols);

  return B_n_rows;
  }



template<typename T1, typename T2>
inline
uword
glue_solve::compute_n_cols(const Glue<T1, T2, glue_solve>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);

  return B_n_cols;
  }
