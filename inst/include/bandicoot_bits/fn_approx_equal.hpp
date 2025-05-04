// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023 Ryan Curtin (http://ratml.org)
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



template<typename T1, typename T2>
coot_warn_unused
inline
bool
approx_equal
  (
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B,
  const char* method,
  const typename T1::pod_type tol
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::pod_type T;

  const char sig = (method != nullptr) ? method[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'r') && (sig != 'b')), "approx_equal(): argument 'method' must be \"absdiff\" or \"reldiff\" or \"both\"" );

  coot_debug_check( (sig == 'b'), "approx_equal(): argument 'method' is \"both\", but only one 'tol' argument has been given" );

  unwrap<T1> UA(A.get_ref());
  unwrap<T2> UB(B.get_ref());

  if (UA.M.n_rows != UB.M.n_rows || UA.M.n_cols != UB.M.n_cols)
    {
    return false;
    }
  else if (UA.M.n_rows == 0 || UA.M.n_cols == 0)
    {
    return true; // empty matrices are equal
    }

  if (sig == 'a')
    {
    return coot_rt_t::approx_equal(UA.get_dev_mem(false),
                                   UA.get_row_offset(),
                                   UA.get_col_offset(),
                                   UA.get_M_n_rows(),
                                   UB.get_dev_mem(false),
                                   UB.get_row_offset(),
                                   UB.get_col_offset(),
                                   UB.get_M_n_rows(),
                                   UA.M.n_rows,
                                   UA.M.n_cols,
                                   sig,
                                   tol,
                                   T(0));
    }
  else
    {
    return coot_rt_t::approx_equal(UA.get_dev_mem(false),
                                   UA.get_row_offset(),
                                   UA.get_col_offset(),
                                   UA.get_M_n_rows(),
                                   UB.get_dev_mem(false),
                                   UB.get_row_offset(),
                                   UB.get_col_offset(),
                                   UB.get_M_n_rows(),
                                   UA.M.n_rows,
                                   UA.M.n_cols,
                                   sig,
                                   T(0),
                                   tol);

    }
  }



template<typename T1, typename T2>
coot_warn_unused
inline
bool
approx_equal
  (
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B,
  const char* method,
  const typename T1::pod_type abs_tol,
  const typename T1::pod_type rel_tol
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (method != nullptr) ? method[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'r') && (sig != 'b')), "approx_equal(): argument 'method' must be \"absdiff\" or \"reldiff\" or \"both\"" );

  unwrap<T1> UA(A.get_ref());
  unwrap<T2> UB(B.get_ref());

  if (UA.M.n_rows != UB.M.n_rows || UA.M.n_cols != UB.M.n_cols)
    {
    return false;
    }
  else if (UA.M.n_rows == 0 || UA.M.n_cols == 0)
    {
    return true; // empty matrices are equal
    }

  return coot_rt_t::approx_equal(UA.get_dev_mem(false),
                                 UA.get_row_offset(),
                                 UA.get_col_offset(),
                                 UA.get_M_n_rows(),
                                 UB.get_dev_mem(false),
                                 UB.get_row_offset(),
                                 UB.get_col_offset(),
                                 UB.get_M_n_rows(),
                                 UA.M.n_rows,
                                 UA.M.n_cols,
                                 sig,
                                 abs_tol,
                                 rel_tol);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
bool
approx_equal
  (
  const BaseCube<typename T1::elem_type, T1>& A,
  const BaseCube<typename T1::elem_type, T2>& B,
  const char* method,
  const typename T1::pod_type tol
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::pod_type T;

  const char sig = (method != nullptr) ? method[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'r') && (sig != 'b')), "approx_equal(): argument 'method' must be \"absdiff\" or \"reldiff\" or \"both\"" );

  coot_debug_check( (sig == 'b'), "approx_equal(): argument 'method' is \"both\", but only one 'tol' argument has been given" );

  unwrap_cube<T1> UA(A.get_ref());
  unwrap_cube<T2> UB(B.get_ref());

  if (UA.M.n_rows != UB.M.n_rows || UA.M.n_cols != UB.M.n_cols || UA.M.n_slices != UB.M.n_slices)
    {
    return false;
    }
  else if (UA.M.n_rows == 0 || UA.M.n_cols == 0 || UA.M.n_slices == 0)
    {
    return true; // empty cubes are equal
    }

  if (sig == 'a')
    {
    return coot_rt_t::approx_equal_cube(UA.get_dev_mem(false),
                                        UA.get_row_offset(),
                                        UA.get_col_offset(),
                                        UA.get_slice_offset(),
                                        UA.get_M_n_rows(),
                                        UA.get_M_n_cols(),
                                        UB.get_dev_mem(false),
                                        UB.get_row_offset(),
                                        UB.get_col_offset(),
                                        UB.get_slice_offset(),
                                        UB.get_M_n_rows(),
                                        UB.get_M_n_cols(),
                                        UA.M.n_rows,
                                        UA.M.n_cols,
                                        UA.M.n_slices,
                                        sig,
                                        tol,
                                        T(0));
    }
  else
    {
    return coot_rt_t::approx_equal_cube(UA.get_dev_mem(false),
                                        UA.get_row_offset(),
                                        UA.get_col_offset(),
                                        UA.get_slice_offset(),
                                        UA.get_M_n_rows(),
                                        UA.get_M_n_cols(),
                                        UB.get_dev_mem(false),
                                        UB.get_row_offset(),
                                        UB.get_col_offset(),
                                        UB.get_slice_offset(),
                                        UB.get_M_n_rows(),
                                        UB.get_M_n_cols(),
                                        UA.M.n_rows,
                                        UA.M.n_cols,
                                        UA.M.n_slices,
                                        sig,
                                        T(0),
                                        tol);

    }
  }



template<typename T1, typename T2>
coot_warn_unused
inline
bool
approx_equal
  (
  const BaseCube<typename T1::elem_type, T1>& A,
  const BaseCube<typename T1::elem_type, T2>& B,
  const char* method,
  const typename T1::pod_type abs_tol,
  const typename T1::pod_type rel_tol
  )
  {
  coot_extra_debug_sigprint();

  const char sig = (method != nullptr) ? method[0] : char(0);

  coot_debug_check( ((sig != 'a') && (sig != 'r') && (sig != 'b')), "approx_equal(): argument 'method' must be \"absdiff\" or \"reldiff\" or \"both\"" );

  unwrap_cube<T1> UA(A.get_ref());
  unwrap_cube<T2> UB(B.get_ref());

  if (UA.M.n_rows != UB.M.n_rows || UA.M.n_cols != UB.M.n_cols || UA.M.n_slices != UB.M.n_slices)
    {
    return false;
    }
  else if (UA.M.n_rows == 0 || UA.M.n_cols == 0 || UA.M.n_slices == 0)
    {
    return true; // empty matrices are equal
    }

  return coot_rt_t::approx_equal_cube(UA.get_dev_mem(false),
                                      UA.get_row_offset(),
                                      UA.get_col_offset(),
                                      UA.get_slice_offset(),
                                      UA.get_M_n_rows(),
                                      UA.get_M_n_cols(),
                                      UB.get_dev_mem(false),
                                      UB.get_row_offset(),
                                      UB.get_col_offset(),
                                      UB.get_slice_offset(),
                                      UB.get_M_n_rows(),
                                      UB.get_M_n_cols(),
                                      UA.M.n_rows,
                                      UA.M.n_cols,
                                      UA.M.n_slices,
                                      sig,
                                      abs_tol,
                                      rel_tol);
  }
