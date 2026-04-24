// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT2, typename T1>
inline
void
op_pinv::apply(Mat<eT2>& out, const Op<T1, op_pinv>& in)
  {
  coot_debug_sigprint();

  const typename T1::elem_type tol = in.aux_a;

  const std::tuple<bool, std::string> result = apply_direct(out, in.m, tol);
  if (std::get<0>(result) == false)
    {
    coot_stop_runtime_error("pinv(): " + std::get<1>(result));
    }
  }



template<typename eT2, typename T1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct(Mat<eT2>& out, const T1& in, const typename T1::elem_type tol)
  {
  coot_debug_sigprint();

  // If `in` is a diagmat():
  //    stored in a vector: apply_direct_diag_vec()
  //    stored in a matrix: apply_direct_diag()
  // if `in` is symmetric:
  //    apply_direct_sym()
  // else:
  //    apply_direct_gen()
  if (resolves_to_diagmat<T1>::value)
    {
    // Now detect whether it is stored in a vector or matrix.
    strip_diagmat<T1> S(in);
    typedef typename strip_diagmat<T1>::stored_type ST1;
    unwrap<ST1> U(S.M);
    if (U.M.n_rows == 1 || U.M.n_cols == 1)
      {
      // Aliases must be handled via a temporary.
      alias_wrapper<Mat<eT2>, typename unwrap<ST1>::stored_type> W(out, U.M);
      return apply_direct_diag(W.use, U.M, tol);
      }
    else
      {
      // Extract the diagonal into a standalone vector for easier processing.
      // Note that aliases don't need to be handled since we are not operating on `in` now.
      Col<typename T1::elem_type> diag = U.M.diag();

      return apply_direct_diag(out, diag, tol);
      }
    }
  else if (resolves_to_symmat<T1>::value)
    {
    // TODO: would be great to avoid actually materializing everything here
    // TODO: a `strip_symmat` struct would be useful for this

    unwrap<T1> U(in);
    extract_subview<typename unwrap<T1>::stored_type> E(U.M);
    // apply_direct_sym() is destructive to the input matrix, so, we may need to make a copy.
    if (is_Mat<T1>::value)
      {
      Mat<typename T1::elem_type> tmp(E.M);
      return apply_direct_sym(out, tmp, tol);
      }
    else
      {
      // We have already created a temporary for unwrapping, so we can destructively use that.
      return apply_direct_sym(out, const_cast<Mat<typename T1::elem_type>&>(E.M), tol);
      }
    }
  else
    {
    unwrap<T1> U(in);
    extract_subview<typename unwrap<T1>::stored_type> E(U.M);
    // apply_direct_gen() is destructive to the input matrix, so, we may need to make a copy.
    if (is_Mat<T1>::value)
      {
      Mat<typename T1::elem_type> tmp(E.M);
      return apply_direct_gen(out, tmp, tol);
      }
    else
      {
      // We have already created a temporary for unwrapping, so we can destructively use that.
      return apply_direct_gen(out, const_cast<Mat<typename T1::elem_type>&>(E.M), tol);
      }
    }
  }



template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_diag(Mat<eT>& out, const Mat<eT>& in, const eT tol)
  {
  coot_debug_sigprint();

  coot_conform_check(in.n_rows != 1 && in.n_cols != 1, "op_pinv::apply_direct_diag_vec(): given input is not a vector (internal error)");

  if (in.n_rows == 0 || in.n_cols == 0)
    {
    // Nothing to do.
    out.reset();
    return std::make_tuple(true, "");
    }

  const uword N = (std::max)(in.n_rows, in.n_cols);

  out.zeros(N, N);

  // Check for any NaNs in the input.
  const bool has_nans = coot_rt_t::any_vec(in.get_dev_mem(false), in.n_elem, (eT) 0, oneway_real_kernel_id::rel_any_nan, oneway_real_kernel_id::rel_any_nan_small);

  if (has_nans == true)
    {
    out.reset();
    return std::make_tuple(false, "NaNs detected in input matrix");
    }

  // Find the values that are below tolerance.
  Mat<eT> abs_in = abs(in);

  // Compute tolerance if not given.
  eT tol_use = tol;
  if (tol == (eT) 0)
    {
    const eT max_val = coot_rt_t::max_vec(abs_in.get_dev_mem(false), abs_in.n_elem);
    tol_use = abs_in.n_elem * max_val * Datum<eT>::eps;
    }

  Mat<uword> tol_indicator(in.n_rows, in.n_cols);
  const mtOp<uword, Mat<eT>, mtop_rel_core<mtop_rel_gt_post>> M(abs_in, (eT) tol_use);
  const mtOp<eT, mtOp<uword, Mat<eT>, mtop_rel_core<mtop_rel_gt_post>>, mtop_conv_to> M2(M);

  // Now invert the diagonal.  Any zero values need to changed to 1, so as to not produce infs or nans.
  // After that, we have to zero out any values below the tolerance.
  Mat<eT> out_vec(abs_in.n_rows, abs_in.n_cols);
  const eOp<Mat<eT>, eop_replace> E(in, 'j', (eT) 0.0, (eT) 1.0);
  const eOp<eOp<Mat<eT>, eop_replace>, eop_scalar_div_pre> E2(E, (eT) 1);
  const eGlue<eOp<eOp<Mat<eT>, eop_replace>, eop_scalar_div_pre>, mtOp<eT, mtOp<uword, Mat<eT>, mtop_rel_core<mtop_rel_gt_post>>, mtop_conv_to>, eglue_schur> E4(E2, M2);
  coot_rt_t::copy(make_proxy(out_vec), make_proxy(E4));

  // Now set the diagonal of the other matrix.
  out.diag() = out_vec;

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_diag(Mat<eT2>& out, const Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_debug_sigprint();
  coot_ignore(junk);

  coot_conform_check(in.n_rows != 1 && in.n_cols != 1, "op_pinv::apply_direct_diag_vec(): given input is not a vector (internal error)");

  if (in.n_rows == 0 || in.n_cols == 0)
    {
    // Nothing to do.
    out.reset();
    return std::make_tuple(true, "");
    }

  const uword N = (std::max)(in.n_rows, in.n_cols);

  out.zeros(N, N);

  // Check for any NaNs in the input.
  const bool has_nans = coot_rt_t::any_vec(in.get_dev_mem(false), in.n_elem, (eT1) 0, oneway_real_kernel_id::rel_any_nan, oneway_real_kernel_id::rel_any_nan_small);

  if (has_nans == true)
    {
    out.reset();
    return std::make_tuple(false, "NaNs detected in input matrix");
    }

  // Find the values that are below tolerance.
  Mat<eT1> abs_in = abs(in);

  // Compute tolerance if not given.
  eT1 tol_use = tol;
  if (tol == (eT1) 0)
    {
    const eT1 max_val = coot_rt_t::max_vec(abs_in.get_dev_mem(false), abs_in.n_elem);
    tol_use = abs_in.n_elem * max_val * Datum<eT1>::eps;
    }

  Mat<uword> tol_indicator(in.n_rows, in.n_cols);
  const mtOp<uword, Mat<eT1>, mtop_rel_core<mtop_rel_gt_post>> M(abs_in, (eT1) tol_use);
  const mtOp<eT1, mtOp<uword, Mat<eT1>, mtop_rel_core<mtop_rel_gt_post>>, mtop_conv_to> M2(M);

  // Now invert the diagonal.  Any zero values need to changed to 1, so as to not produce infs or nans.
  Mat<eT1> out_vec(abs_in.n_rows, abs_in.n_cols);
  const eOp<Mat<eT1>, eop_replace> E(in, 'j', (eT1) 0.0, (eT1) 1.0);
  const eOp<eOp<Mat<eT1>, eop_replace>, eop_scalar_div_pre> E2(E, (eT1) 1);
  const eGlue<eOp<eOp<Mat<eT1>, eop_replace>, eop_scalar_div_pre>, mtOp<eT1, mtOp<uword, Mat<eT1>, mtop_rel_core<mtop_rel_gt_post>>, mtop_conv_to>, eglue_schur> E3(E2, M2);
  coot_rt_t::copy(make_proxy(out_vec), make_proxy(E3));

  // Now set the diagonal of the other matrix.  This also performs the conversion.
  diagview<eT2> out_d(out, 0, 0, (std::min)(out.n_rows, out.n_cols));
  coot_rt_t::copy(make_proxy(out_d), make_proxy_col(out_vec));

  return std::make_tuple(true, "");
  }



template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_sym(Mat<eT>& out, Mat<eT>& in, const eT tol)
  {
  coot_debug_sigprint();

  if (in.n_rows == 0 || in.n_cols == 0)
    {
    // Nothing to do.
    out.reset();
    return std::make_tuple(true, "");
    }

  Col<eT> eigvals(in.n_rows);

  //
  // Step 1. compute eigendecomposition, sorting eigenvalues descending by absolute value.
  //

  // `in` will store the eigenvectors after this call (destructive).
  const std::tuple<bool, std::string> result = coot_rt_t::eig_sym(in.get_dev_mem(true), in.n_rows, true, eigvals.get_dev_mem(true));
  if (std::get<0>(result) == false)
    {
    out.reset();
    return std::make_tuple(false, "eigendecomposition failed");
    }

  Col<eT> abs_eigvals = abs(eigvals);

  Col<uword> eigval_order(in.n_rows);
  // This also sorts `abs_eigvals`.
  coot_rt_t::sort_index_vec(eigval_order.get_dev_mem(false), abs_eigvals.get_dev_mem(false), abs_eigvals.n_elem, 1 /* descending */, 0);

  //
  // Step 2. keep all eigenvalues greater than the tolerance.
  //
  const eT tol_use = (tol == eT(0)) ? in.n_rows * abs_eigvals[0] * Datum<eT>::eps : tol;

  Col<uword> tol_indicators(eigval_order.n_elem);
  const mtOp<uword, Col<eT>, mtop_rel_core<mtop_rel_gt_post>> M(abs_eigvals, tol_use);
  coot_rt_t::copy(make_proxy(tol_indicators), make_proxy(M));
  const uword num_eigvals = coot_rt_t::accu(tol_indicators.get_dev_mem(false), eigval_order.n_elem);
  if (num_eigvals == 0)
    {
    out.zeros(in.n_rows, in.n_cols);
    return std::make_tuple(true, "");
    }

  // Filter the top eigenvalues and eigenvectors.
  Col<eT> filtered_eigvals(num_eigvals);
  Mat<eT> filtered_eigvecs(in.n_rows, num_eigvals);
  coot_rt_t::reorder_cols(filtered_eigvals.get_dev_mem(false), eigvals.get_dev_mem(false), 1, eigval_order.get_dev_mem(false), num_eigvals);
  coot_rt_t::reorder_cols(filtered_eigvecs.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, eigval_order.get_dev_mem(false), num_eigvals);

  //
  // 3. Invert the eigenvalues we kept.  Avoid divergence by replacing 0s with 1s.
  //
  const eOp<Col<eT>, eop_replace> E(filtered_eigvals, 'j', (eT) 0, (eT) 1);
  const eOp<eOp<Col<eT>, eop_replace>, eop_scalar_div_pre> E2(E, (eT) 1);
  coot_rt_t::copy(make_proxy(filtered_eigvals), make_proxy(E2));

  //
  // 4. Construct output.
  //
  out.set_size(filtered_eigvecs.n_rows, filtered_eigvecs.n_rows);
  Mat<eT> tmp(filtered_eigvecs.n_rows, filtered_eigvecs.n_cols);
  // tmp = filtered_eigvecs * diagmat(inverted eigvals)
  coot_rt_t::mul_diag(tmp.get_dev_mem(false), tmp.n_rows, tmp.n_cols,
                      (eT) 1, filtered_eigvecs.get_dev_mem(false), false, false, 0, 0, filtered_eigvecs.n_rows,
                      filtered_eigvals.get_dev_mem(false), true /* diag */, false, 0, 0, 1);
  // out = tmp * filtered_eigvecs.t()
  coot_rt_t::gemm<eT, false, true>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                   tmp.get_dev_mem(true), tmp.n_rows, tmp.n_cols,
                                   filtered_eigvecs.get_dev_mem(true), (eT) 1.0, (eT) 0.0,
                                   0, 0, out.n_rows,
                                   0, 0, tmp.n_rows,
                                   0, 0, filtered_eigvecs.n_rows);

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_sym(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_debug_sigprint();
  coot_ignore(junk);

  // We need to perform this into a temporary, and then convert.
  Mat<eT1> tmp;
  const std::tuple<bool, std::string> status = apply_direct_sym(tmp, in, tol);
  if (std::get<0>(status) == false)
    {
    return status;
    }

  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy(make_proxy(out), make_proxy(tmp));
  return status; // (true, "")
  }




template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_gen(Mat<eT>& out, Mat<eT>& in, const eT tol)
  {
  coot_debug_sigprint();

  if (in.n_rows == 0 || in.n_cols == 0)
    {
    // Nothing to do.
    out.reset();
    return std::make_tuple(true, "");
    }

  //
  // 1. Transpose input if needed so that n_rows >= n_cols.
  //
  Mat<eT> tmp_in;
  Mat<eT>& in_use = (in.n_rows < in.n_cols) ? tmp_in : in;
  if (in.n_rows < in.n_cols)
    {
    tmp_in.set_size(in.n_cols, in.n_rows);
    coot_rt_t::copy(make_proxy(tmp_in), make_proxy(in.t()));
    }

  //
  // 2. Compute the SVD.
  //
  Mat<eT> U(in_use.n_rows, in_use.n_rows);
  Col<eT> S((std::min)(in_use.n_rows, in_use.n_cols));
  Mat<eT> V(in_use.n_cols, in_use.n_cols);

  const std::tuple<bool, std::string> status = coot_rt_t::svd(U.get_dev_mem(true),
                                                              S.get_dev_mem(true),
                                                              V.get_dev_mem(true),
                                                              in_use.get_dev_mem(true),
                                                              in_use.n_rows,
                                                              in_use.n_cols,
                                                              true);

  if (std::get<0>(status) == false)
    {
    return std::make_tuple(false, "SVD failed");
    }

  //
  // 2. Compute tolerance.  Note that the singular values are returned in descending order already.
  //
  const eT largest_sv = S[0];
  const eT tol_use = (tol == eT(0)) ? in_use.n_rows * largest_sv * Datum<eT>::eps : tol;

  //
  // 3. Keep singular values that are greater than the tolerance.
  //
  Col<uword> S_above_tol(S.n_elem);
  const mtOp<uword, Col<eT>, mtop_rel_core<mtop_rel_gteq_post>> M(S, tol_use);
  coot_rt_t::copy(make_proxy(S_above_tol), make_proxy(M));
  const uword num_svs = coot_rt_t::accu(S_above_tol.get_dev_mem(false), S_above_tol.n_elem);
  if (num_svs == 0)
    {
    out.zeros(in.n_rows, in.n_cols);
    return std::make_tuple(true, "");
    }

  // Create aliases for the filtered left/right singular vectors and filtered singular values.
  Mat<eT> filtered_U(U.get_dev_mem(false), U.n_rows, num_svs);
  Mat<eT> filtered_S(S.get_dev_mem(false), num_svs, 1);
  // Unfortunately filtering V means shedding rows, not columns.
  Mat<eT> filtered_V;
  if (num_svs != V.n_rows)
    {
    filtered_V = V.rows(0, num_svs - 1);
    }
  else
    {
    filtered_V = Mat<eT>(V.get_dev_mem(false), num_svs, V.n_cols);
    }

  //
  // 4. Invert singular values.  Avoid divergence by replacing 0s with 1s.
  //
  const eOp<Mat<eT>, eop_replace> E(filtered_S, 'j', (eT) 0, (eT) 1);
  const eOp<eOp<Mat<eT>, eop_replace>, eop_scalar_div_pre> E2(E, (eT) 1);
  coot_rt_t::copy(make_proxy(filtered_S), make_proxy(E2));

  //
  // 5. Reconstruct as subset of V * diagmat(inv_s) * subset of U
  //    (transposed if n_rows < n_cols).
  //
  if (in.n_rows < in.n_cols)
    {
    // U' = U * diagmat(s)   (in-place into U)
    coot_rt_t::mul_diag(filtered_U.get_dev_mem(false), filtered_U.n_rows, filtered_U.n_cols,
                        (eT) 1.0, filtered_U.get_dev_mem(false), false, false, 0, 0, filtered_U.n_rows,
                        filtered_S.get_dev_mem(false), true, false, 0, 0, 1);

    // out = U' * V^T (remember V is already transposed)
    out.set_size(filtered_U.n_rows, filtered_V.n_rows);
    coot_rt_t::gemm<eT, false, false>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                      filtered_U.get_dev_mem(true), filtered_U.n_rows, filtered_U.n_cols,
                                      filtered_V.get_dev_mem(true), (eT) 1.0, (eT) 0.0,
                                      0, 0, out.n_rows,
                                      0, 0, filtered_U.n_rows,
                                      0, 0, filtered_V.n_rows);
    }
  else
    {
    // tmp = V * diagmat(s)   (in-place into V)
    Mat<eT> tmp(filtered_V.n_cols, filtered_V.n_rows);
    coot_rt_t::mul_diag(tmp.get_dev_mem(false), tmp.n_rows, tmp.n_cols,
                        (eT) 1.0, filtered_V.get_dev_mem(false), false, true, 0, 0, filtered_V.n_rows,
                        filtered_S.get_dev_mem(false), true, false, 0, 0, 1);

    // out = tmp * U^T
    out.set_size(tmp.n_rows, filtered_U.n_rows);
    coot_rt_t::gemm<eT, false, true>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                     tmp.get_dev_mem(true), tmp.n_rows, tmp.n_cols,
                                     filtered_U.get_dev_mem(true), (eT) 1.0, (eT) 0.0,
                                     0, 0, out.n_rows,
                                     0, 0, tmp.n_rows,
                                     0, 0, filtered_U.n_rows);
    }

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_gen(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_debug_sigprint();
  coot_ignore(junk);

  // We need to perform this into a temporary, and then convert.
  Mat<eT1> tmp;
  const std::tuple<bool, std::string> status = apply_direct_gen(tmp, in, tol);
  if (std::get<0>(status) == false)
    {
    return status;
    }

  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy(make_proxy(out), make_proxy(tmp));
  return status; // (true, "")
  }



template<typename T1>
inline
uword
op_pinv::compute_n_rows(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_pinv::compute_n_cols(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
