// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2023 Conrad Sanderson (https://conradsanderson.id.au)
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
bool
svd
  (
         Col<typename T1::pod_type>&     S,
  const Base<typename T1::elem_type,T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  // The SVD implementation will use/destroy the memory of A, but we may need to transpose the operation depending on the size.
  SizeProxy<T1> P(X.get_ref());

  if (P.get_n_elem() == 0)
    {
    S.reset();
    return true;
    }

  const uword in_n_rows = P.get_n_rows();
  const uword in_n_cols = P.get_n_cols();

  // Initialize temporary matrices for work.
  // This assumes that the actual gesvd implementation will not reference U or V if we ask to not compute them.
  // (That's at least true for CPU LAPACK...)
  Mat<eT> U(1, 1);
  Mat<eT> V(1, 1);

  // The svd() function expects that n_rows >= n_cols; so, if that's not the case,
  // we will just transpose A.

  Mat<eT> A;
  if (in_n_rows >= in_n_cols)
    {
    A = Mat<eT>(X.get_ref());
    }
  else
    {
    A = Mat<eT>(htrans(X.get_ref()));
    }

  // This will be the right size regardless of whether we needed to transpose:
  // the number of singular values is equal to min(n_rows, n_cols).
  S.set_size(A.n_cols);

  const std::tuple<bool, std::string>& result = coot_rt_t::svd(U.get_dev_mem(true),
                                                               S.get_dev_mem(true),
                                                               V.get_dev_mem(true),
                                                               A.get_dev_mem(true),
                                                               A.n_rows,
                                                               A.n_cols,
                                                               false);

  return std::get<0>(result);
  }



template<typename T1>
coot_warn_unused
inline
Col<typename T1::pod_type>
svd
  (
  const Base<typename T1::elem_type,T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::pod_type   T;

  Col<T> S;
  typedef typename T1::elem_type eT;

  // The SVD implementation will use/destroy the memory of A, but we may need to transpose the operation depending on the size.
  SizeProxy<T1> P(X.get_ref());

  if (P.get_n_elem() == 0)
    {
    S.reset();
    return S;
    }

  const uword in_n_rows = P.get_n_rows();
  const uword in_n_cols = P.get_n_cols();

  // Initialize temporary matrices for work.
  // This assumes that the actual gesvd implementation will not reference U or V if we ask to not compute them.
  // (That's at least true for CPU LAPACK...)
  Mat<eT> U(1, 1);
  Mat<eT> V(1, 1);

  // The svd() function expects that n_rows >= n_cols; so, if that's not the case,
  // we will just transpose A.

  Mat<eT> A;
  if (in_n_rows >= in_n_cols)
    {
    A = Mat<eT>(X.get_ref());
    }
  else
    {
    A = Mat<eT>(htrans(X.get_ref()));
    }

  // This will be the right size regardless of whether we needed to transpose:
  // the number of singular values is equal to min(n_rows, n_cols).
  S.set_size(A.n_cols);

  const std::tuple<bool, std::string>& result = coot_rt_t::svd(U.get_dev_mem(true),
                                                               S.get_dev_mem(true),
                                                               V.get_dev_mem(true),
                                                               A.get_dev_mem(true),
                                                               A.n_rows,
                                                               A.n_cols,
                                                               false);

  if (std::get<0>(result) == false)
    {
    S.reset();
    coot_stop_runtime_error("coot::svd(): decomposition failed: " + std::get<1>(result));
    }

  return S;
  }



// only "std" method is supported for now!
template<typename T1>
inline
bool
svd
  (
         Mat<typename T1::elem_type>&    U,
         Col<typename T1::pod_type >&    S,
         Mat<typename T1::elem_type>&    V,
  const Base<typename T1::elem_type,T1>& X,
  const char*                            method = "std", // differs from Armadillo
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  coot_debug_check
    (
    ( ((void*)(&U) == (void*)(&S)) || (&U == &V) || ((void*)(&S) == (void*)(&V)) ),
    "svd(): two or more output objects are the same object"
    );

  const char sig = (method != nullptr) ? method[0] : char(0);

  coot_debug_check( ((sig != 's') && (sig != 'd')), "svd(): unknown method specified" );

  if (sig == 'd')
    {
    coot_debug_warn_level(2, "svd(): \"dc\" method not supported by bandicoot; falling back to \"std\"");
    }

  SizeProxy<T1> P(X.get_ref());

  if (P.get_n_elem() == 0)
    {
    U.reset();
    S.reset();
    V.reset();
    return true;
    }

  const uword in_n_rows = P.get_n_rows();
  const uword in_n_cols = P.get_n_cols();

  // The svd() function expects that n_rows >= n_cols; so, if that's not the case,
  // we will just transpose A.  This will mean we have to do a little postprocessing
  // of the results, though.

  U.set_size(in_n_rows, in_n_rows);
  S.set_size(std::min(in_n_rows, in_n_cols));
  V.set_size(in_n_cols, in_n_cols);

  if (in_n_rows >= in_n_cols)
    {
    Mat<eT> A(X.get_ref());

    const std::tuple<bool, std::string>& status = coot_rt_t::svd(U.get_dev_mem(true),
                                                                 S.get_dev_mem(true),
                                                                 V.get_dev_mem(true),
                                                                 A.get_dev_mem(true),
                                                                 A.n_rows,
                                                                 A.n_cols,
                                                                 true);

    if(std::get<0>(status) == false)
      {
      U.reset();
      S.reset();
      V.reset();
      }
    else
      {
      // We got back V^*, so we have to transpose the result.
      V = htrans(V);
      }

    return std::get<0>(status);
    }
  else
    {
    Mat<eT> A(htrans(X.get_ref()));

    // Note that V and U are swapped here!
    const std::tuple<bool, std::string>& status = coot_rt_t::svd(V.get_dev_mem(true),
                                                                 S.get_dev_mem(true),
                                                                 U.get_dev_mem(true),
                                                                 A.get_dev_mem(true),
                                                                 A.n_rows,
                                                                 A.n_cols,
                                                                 true);

    if(std::get<0>(status) == false)
      {
      U.reset();
      S.reset();
      V.reset();
      }
    else
      {
      // We got back U^*, so we have to transpose the result.
      U = htrans(U);
      }

    return std::get<0>(status);
    }
  }
