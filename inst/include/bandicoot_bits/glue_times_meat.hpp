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



template<uword N>
template<typename out_eT, typename T1, typename T2>
inline
void
glue_times_redirect<N>::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X)
  {
  coot_extra_debug_sigprint();

  const partial_unwrap<T1> tmp1(X.A);
  const partial_unwrap<T2> tmp2(X.B);

  const typename partial_unwrap<T1>::stored_type& A = tmp1.M;
  const typename partial_unwrap<T2>::stored_type& B = tmp2.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val()) : out_eT(0);

  const bool alias = tmp1.is_alias(out) || tmp2.is_alias(out);

  if(alias == false)
    {
    glue_times::apply
      <
      out_eT,
      T1,
      T2,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times)
      >
      (out, A, B, alpha);
    }
  else
    {
    Mat<out_eT> tmp;

    glue_times::apply
      <
      out_eT,
      T1,
      T2,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times)
      >
      (tmp, A, B, alpha);

    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename T1, typename T2>
inline
void
glue_times_redirect<2>::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X)
  {
  coot_extra_debug_sigprint();

  const partial_unwrap<T1> tmp1(X.A);
  const partial_unwrap<T2> tmp2(X.B);

  const typename partial_unwrap<T1>::stored_type& A = tmp1.M;
  const typename partial_unwrap<T2>::stored_type& B = tmp2.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val()) : out_eT(0);

  const bool alias = tmp1.is_alias(out) || tmp2.is_alias(out);

  if(alias == false)
    {
    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times)
      >
      (out, A, B, alpha);
    }
  else
    {
    Mat<out_eT> tmp;

    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times)
      >
      (tmp, A, B, alpha);

    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename T1, typename T2, typename T3>
inline
void
glue_times_redirect<3>::apply(Mat<out_eT>& out, const Glue<Glue<T1, T2, glue_times>, T3, glue_times>& X)
  {
  coot_extra_debug_sigprint();

  // we have exactly 3 objects
  // hence we can safely expand X as X.A.A, X.A.B and X.B

  const partial_unwrap<T1> tmp1(X.A.A);
  const partial_unwrap<T2> tmp2(X.A.B);
  const partial_unwrap<T3> tmp3(X.B  );

  const typename partial_unwrap<T1>::stored_type& A = tmp1.M;
  const typename partial_unwrap<T2>::stored_type& B = tmp2.M;
  const typename partial_unwrap<T3>::stored_type& C = tmp3.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val() * tmp3.get_val()) : out_eT(0);

  const bool alias = tmp1.is_alias(out) || tmp2.is_alias(out) || tmp3.is_alias(out);

  if(alias == false)
    {
    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      typename partial_unwrap<T3>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      partial_unwrap<T3>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times)
      >
      (out, A, B, C, alpha);
    }
  else
    {
    Mat<out_eT> tmp;

    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      typename partial_unwrap<T3>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      partial_unwrap<T3>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times)
      >
      (tmp, A, B, C, alpha);

    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename T1, typename T2, typename T3, typename T4>
inline
void
glue_times_redirect<4>::apply(Mat<out_eT>& out, const Glue<Glue<Glue<T1, T2, glue_times>, T3, glue_times>, T4, glue_times>& X)
  {
  coot_extra_debug_sigprint();

  // there is exactly 4 objects
  // hence we can safely expand X as X.A.A.A, X.A.A.B, X.A.B and X.B

  const partial_unwrap<T1> tmp1(X.A.A.A);
  const partial_unwrap<T2> tmp2(X.A.A.B);
  const partial_unwrap<T3> tmp3(X.A.B  );
  const partial_unwrap<T4> tmp4(X.B    );

  const typename partial_unwrap<T1>::stored_type& A = tmp1.M;
  const typename partial_unwrap<T2>::stored_type& B = tmp2.M;
  const typename partial_unwrap<T3>::stored_type& C = tmp3.M;
  const typename partial_unwrap<T4>::stored_type& D = tmp4.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times || partial_unwrap<T4>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val() * tmp3.get_val() * tmp4.get_val()) : out_eT(0);

  const bool alias = tmp1.is_alias(out) || tmp2.is_alias(out) || tmp3.is_alias(out) || tmp4.is_alias(out);

  if(alias == false)
    {
    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      typename partial_unwrap<T3>::stored_type,
      typename partial_unwrap<T4>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      partial_unwrap<T3>::do_trans,
      partial_unwrap<T4>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times || partial_unwrap<T4>::do_times)
      >
      (out, A, B, C, D, alpha);
    }
  else
    {
    Mat<out_eT> tmp;

    glue_times::apply
      <
      out_eT,
      typename partial_unwrap<T1>::stored_type,
      typename partial_unwrap<T2>::stored_type,
      typename partial_unwrap<T3>::stored_type,
      typename partial_unwrap<T4>::stored_type,
      partial_unwrap<T1>::do_trans,
      partial_unwrap<T2>::do_trans,
      partial_unwrap<T3>::do_trans,
      partial_unwrap<T4>::do_trans,
      (partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times || partial_unwrap<T3>::do_times || partial_unwrap<T4>::do_times)
      >
      (tmp, A, B, C, D, alpha);

    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename T1, typename T2>
inline
void
glue_times::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times>& X)
  {
  coot_extra_debug_sigprint();

  const sword N_mat = 1 + depth_lhs< glue_times, Glue<T1, T2, glue_times> >::num;

  coot_extra_debug_print(coot_str::format("N_mat = %d") % N_mat);

  glue_times_redirect<N_mat>::apply(out, X);
  }



template<typename eT1, typename eT2, const bool do_trans_A, const bool do_trans_B>
inline
uword
glue_times::mul_storage_cost(const Mat<eT1>& A, const Mat<eT2>& B)
  {
  const uword final_A_n_rows = (do_trans_A == false) ? A.n_rows : A.n_cols;
  const uword final_B_n_cols = (do_trans_B == false) ? B.n_cols : B.n_rows;

  return final_A_n_rows * final_B_n_cols;
  }



template
  <
  typename   out_eT,
  typename   T1,
  typename   T2,
  const bool do_trans_A,
  const bool do_trans_B,
  const bool use_alpha
  >
inline
void
glue_times::apply
  (
        Mat<out_eT>& out,
  const T1&          A, // Mat or subview
  const T2&          B,
  const out_eT       alpha
  )
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_trans_mul_size<do_trans_A, do_trans_B>(A.n_rows, A.n_cols, B.n_rows, B.n_cols, "matrix multiplication");

  const uword final_n_rows = (do_trans_A == false) ? A.n_rows : A.n_cols;
  const uword final_n_cols = (do_trans_B == false) ? B.n_cols : B.n_rows;

  out.set_size(final_n_rows, final_n_cols);

  if( (A.n_elem == 0) || (B.n_elem == 0) )
    {
    out.zeros();
    return;
    }


  if( (do_trans_A == false) && (do_trans_B == false) && (use_alpha == false) )
    {
         if( (A.n_rows == 1) && (is_cx<out_eT>::no) )  { gemv<true,         false, false>::apply(out, B, A); }
    else if( (B.n_cols == 1)                        )  { gemv<false,        false, false>::apply(out, A, B); }
    else                                               { gemm<false, false, false, false>::apply(out, A, B); }
    }
  else
  if( (do_trans_A == false) && (do_trans_B == false) && (use_alpha == true) )
    {
         if( (A.n_rows == 1) && (is_cx<out_eT>::no) )  { gemv<true,         true, false>::apply(out, B, A, alpha); }
    else if( (B.n_cols == 1)                        )  { gemv<false,        true, false>::apply(out, A, B, alpha); }
    else                                               { gemm<false, false, true, false>::apply(out, A, B, alpha); }
    }
  else
  if( (do_trans_A == true) && (do_trans_B == false) && (use_alpha == false) )
    {
         if( (A.n_cols == 1) && (is_cx<out_eT>::no) )  { gemv<true,        false, false>::apply(out, B, A); }
    else if( (B.n_cols == 1)                        )  { gemv<true,        false, false>::apply(out, A, B); }
    else                                               { gemm<true, false, false, false>::apply(out, A, B); }
    }
  else
  if( (do_trans_A == true) && (do_trans_B == false) && (use_alpha == true) )
    {
         if( (A.n_cols == 1) && (is_cx<out_eT>::no) )  { gemv<true,        true, false>::apply(out, B, A, alpha); }
    else if( (B.n_cols == 1)                        )  { gemv<true,        true, false>::apply(out, A, B, alpha); }
    else                                               { gemm<true, false, true, false>::apply(out, A, B, alpha); }
    }
  else
  if( (do_trans_A == false) && (do_trans_B == true) && (use_alpha == false) )
    {
         if( (A.n_rows == 1) && (is_cx<out_eT>::no)  )  { gemv<false,       false, false>::apply(out, B, A); }
    else if( (B.n_rows == 1) && (is_cx<out_eT>::no)  )  { gemv<false,       false, false>::apply(out, A, B); }
    else                                                { gemm<false, true, false, false>::apply(out, A, B); }
    }
  else
  if( (do_trans_A == false) && (do_trans_B == true) && (use_alpha == true) )
    {
         if( (A.n_rows == 1) && (is_cx<out_eT>::no) ) { gemv<false,       true, false>::apply(out, B, A, alpha); }
    else if( (B.n_rows == 1) && (is_cx<out_eT>::no) ) { gemv<false,       true, false>::apply(out, A, B, alpha); }
    else                                              { gemm<false, true, true, false>::apply(out, A, B, alpha); }
    }
  else
  if( (do_trans_A == true) && (do_trans_B == true) && (use_alpha == false) )
    {
         if( (A.n_cols == 1) && (is_cx<out_eT>::no) )  { gemv<false,      false, false>::apply(out, B, A); }
    else if( (B.n_rows == 1) && (is_cx<out_eT>::no) )  { gemv<true,       false, false>::apply(out, A, B); }
    else                                               { gemm<true, true, false, false>::apply(out, A, B); }
    }
  else
  if( (do_trans_A == true) && (do_trans_B == true) && (use_alpha == true) )
    {
         if( (A.n_cols == 1) && (is_cx<out_eT>::no) )  { gemv<false,      true, false>::apply(out, B, A, alpha); }
    else if( (B.n_rows == 1) && (is_cx<out_eT>::no) )  { gemv<true,       true, false>::apply(out, A, B, alpha); }
    else                                               { gemm<true, true, true, false>::apply(out, A, B, alpha); }
    }
  }



template
  <
  typename   out_eT,
  typename   T1,
  typename   T2,
  typename   T3,
  const bool do_trans_A,
  const bool do_trans_B,
  const bool do_trans_C,
  const bool use_alpha
  >
inline
void
glue_times::apply
  (
        Mat<out_eT>& out,
  const T1&          A, // Mat or subview
  const T2&          B,
  const T3&          C,
  const out_eT       alpha
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  typedef typename T3::elem_type eT3;

  Mat<out_eT> tmp;

  const uword storage_cost_AB = glue_times::mul_storage_cost<eT1, eT2, do_trans_A, do_trans_B>(A, B);
  const uword storage_cost_BC = glue_times::mul_storage_cost<eT2, eT3, do_trans_B, do_trans_C>(B, C);

  if(storage_cost_AB <= storage_cost_BC)
    {
    // out = (A*B)*C

    glue_times::apply<out_eT, T1, T2, do_trans_A, do_trans_B, use_alpha>(tmp, A,   B, alpha);
    glue_times::apply<out_eT, Mat<out_eT>, T3, false,      do_trans_C, false    >(out, tmp, C, out_eT(0));
    }
  else
    {
    // out = A*(B*C)

    glue_times::apply<out_eT, T2, T3, do_trans_B, do_trans_C, use_alpha>(tmp, B, C,   alpha);
    glue_times::apply<out_eT, T1, Mat<out_eT>, do_trans_A, false,      false    >(out, A, tmp, out_eT(0));
    }
  }



template
  <
  typename   out_eT,
  typename   T1,
  typename   T2,
  typename   T3,
  typename   T4,
  const bool do_trans_A,
  const bool do_trans_B,
  const bool do_trans_C,
  const bool do_trans_D,
  const bool use_alpha
  >
inline
void
glue_times::apply
  (
        Mat<out_eT>& out,
  const T1&          A, // Mat or subview
  const T2&          B,
  const T3&          C,
  const T4&          D,
  const out_eT       alpha
  )
  {
  coot_extra_debug_sigprint();

  Mat<out_eT> tmp;

  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  typedef typename T3::elem_type eT3;
  typedef typename T4::elem_type eT4;

  const uword storage_cost_AC = glue_times::mul_storage_cost<eT1, eT3, do_trans_A, do_trans_C>(A, C);
  const uword storage_cost_BD = glue_times::mul_storage_cost<eT2, eT4, do_trans_B, do_trans_D>(B, D);

  if(storage_cost_AC <= storage_cost_BD)
    {
    // out = (A*B*C)*D

    glue_times::apply<out_eT, T1, T2, T3, do_trans_A, do_trans_B, do_trans_C, use_alpha>(tmp, A, B, C, alpha);

    glue_times::apply<out_eT, T4, Mat<out_eT>, false, do_trans_D, false>(out, tmp, D, out_eT(0));
    }
  else
    {
    // out = A*(B*C*D)

    glue_times::apply<out_eT, T2, T3, T4, do_trans_B, do_trans_C, do_trans_D, use_alpha>(tmp, B, C, D, alpha);

    glue_times::apply<out_eT, T1, Mat<out_eT>, do_trans_A, false, false>(out, A, tmp, out_eT(0));
    }
  }



template<typename T1, typename T2>
inline
uword
glue_times::compute_n_rows(const Glue<T1, T2, glue_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
glue_times::compute_n_cols(const Glue<T1, T2, glue_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  return B_n_cols;
  }



// glue_times_diag

template<typename out_eT, typename T1, typename T2>
inline
void
glue_times_diag::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_times_diag>& X)
  {
  coot_extra_debug_sigprint();

  strip_diagmat<T1> s1(X.A);
  strip_diagmat<T2> s2(X.B);

  // partially unwrap arguments for matrix multiplication
  typedef typename strip_diagmat<T1>::stored_type ST1;
  typedef typename strip_diagmat<T2>::stored_type ST2;
  partial_unwrap<ST1> p1(s1.M);
  partial_unwrap<ST2> p2(s2.M);

  constexpr bool A_diag = s1.do_diagmat;
  constexpr bool B_diag = s2.do_diagmat;
  constexpr bool A_trans = p1.do_trans;
  constexpr bool B_trans = p2.do_trans;
  const uword A_n_elem = p1.M.n_elem;
  const uword A_n_rows = (A_diag) ? A_n_elem : p1.M.n_rows;
  const uword A_n_cols = (A_diag) ? A_n_elem : p1.M.n_cols;
  const uword B_n_elem = p2.M.n_elem;
  const uword B_n_rows = (B_diag) ? B_n_elem : p2.M.n_rows;
  const uword B_n_cols = (B_diag) ? B_n_elem : p2.M.n_cols;
  const uword C_n_rows = (A_trans) ? A_n_cols : A_n_rows;
  const uword C_n_cols = (B_trans) ? B_n_rows : B_n_cols;

  coot_debug_assert_trans_mul_size<A_trans, B_trans>(A_n_rows, A_n_cols, B_n_rows, B_n_cols, "matrix multiplication");

  const out_eT alpha = (p1.get_val() == out_eT(1) && p2.get_val() == out_eT(1)) ? out_eT(1) : p1.get_val() * p2.get_val();

  out.zeros(C_n_rows, C_n_cols);

  if (A_diag && B_diag)
    {
    // diagmat(A) * diagmat(B)

    // In this case, we can do an elementwise multiplication of the diagonal
    // vectors, and create a new diagonal matrix.
    // TODO: once partial_unwrap supports subviews, or even diagviews, we can use subview offsets here
    Col<out_eT> tmp(A_n_elem);
    coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                       tmp.get_dev_mem(false),
                       p1.M.get_dev_mem(false),
                       p2.M.get_dev_mem(false),
                       tmp.n_rows, tmp.n_cols,
                       0, 0, tmp.n_rows,
                       0, 0, p1.M.n_rows,
                       0, 0, p2.M.n_rows);
    if (alpha != out_eT(1))
      {
      coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                            tmp.get_dev_mem(false), tmp.get_dev_mem(false),
                            alpha, (out_eT) 1,
                            tmp.n_rows, tmp.n_cols,
                            0, 0, tmp.n_rows,
                            0, 0, tmp.n_rows);
      }

    // Set the diagonal of `out` to the vector in `tmp`.
    coot_rt_t::copy_mat(out.get_dev_mem(false), tmp.get_dev_mem(false),
                        1, A_n_elem,
                        0, 0, C_n_rows + 1,
                        0, 0, 1);
    }
  else if (!A_diag && !B_diag)
    {
    coot_stop_runtime_error("glue_times_diag::apply(): neither matrix to be multiplied is a diagonal matrix");
    }
  else
    {
    coot_rt_t::mul_diag(out.get_dev_mem(false), C_n_rows, C_n_cols, alpha, p1.M.get_dev_mem(false), A_diag, A_trans, p2.M.get_dev_mem(false), B_diag, B_trans);
    }
  }



template<typename T1, typename T2>
inline
uword
glue_times_diag::compute_n_rows(const Glue<T1, T2, glue_times_diag>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  return A_n_rows;
  }



template<typename T1, typename T2>
inline
uword
glue_times_diag::compute_n_cols(const Glue<T1, T2, glue_times_diag>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  return B_n_cols;
  }
