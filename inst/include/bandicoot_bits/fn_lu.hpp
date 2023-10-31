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
lu
  (
         Mat<typename T1::elem_type>&     L,
         Mat<typename T1::elem_type>&     U,
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (&L) == (&U), "lu(): L and U are the same object" );

  SizeProxy<T1> S(X.get_ref());

  const uword in_n_rows = S.get_n_rows();
  const uword in_n_cols = S.get_n_cols();
  const uword min_rows_cols = std::min(in_n_rows, in_n_cols);

  // If n_rows <= n_cols, then we can do the operation in-place in U.
  // Otherwise, U will be smaller than the matrix we need to do the LU decomposition into.
  bool U_inplace = false;
  Mat<typename T1::elem_type> in; // may or may not be used
  if (in_n_rows > in_n_cols)
    {
    // We need to use `in` separately.
    in = X.get_ref();
    U.set_size(min_rows_cols, in_n_cols);
    }
  else
    {
    // We can use the memory of `U` directly.
    U_inplace = true;
    U = X.get_ref();
    }

  L.set_size(in_n_rows, min_rows_cols);

  if (U.n_elem == 0)
    {
    // Nothing to do---leave early.
    L.set_size(in_n_rows, 0);
    U.set_size(0, in_n_cols);
    return true;
    }

  const std::tuple<bool, std::string> result = coot_rt_t::lu(L.get_dev_mem(true),
                                                             U.get_dev_mem(true),
                                                             (U_inplace ? U.get_dev_mem(true) : in.get_dev_mem(true)),
                                                             false /* no pivoting */,
                                                             U.get_dev_mem(false) /* ignored */,
                                                             in_n_rows,
                                                             in_n_cols);
  if (!std::get<0>(result))
    {
    coot_debug_warn_level(3, "lu(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }



template<typename T1>
inline
bool
lu
  (
         Mat<typename T1::elem_type>&     L,
         Mat<typename T1::elem_type>&     U,
         Mat<typename T1::elem_type>&     P,
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (&L) == (&U), "lu(): L and U are the same object" );

  SizeProxy<T1> S(X.get_ref());

  const uword in_n_rows = S.get_n_rows();
  const uword in_n_cols = S.get_n_cols();
  const uword min_rows_cols = std::min(in_n_rows, in_n_cols);

  // If n_rows <= n_cols, then we can do the operation in-place in U.
  // Otherwise, U will be smaller than the matrix we need to do the LU decomposition into.
  bool U_inplace = false;
  Mat<typename T1::elem_type> in; // may or may not be used
  if (in_n_rows > in_n_cols)
    {
    // We need to use `in` separately.
    in = X.get_ref();
    U.set_size(min_rows_cols, in_n_cols);
    }
  else
    {
    // We can use the memory of `U` directly.
    U_inplace = true;
    U = X.get_ref();
    }

  L.set_size(in_n_rows, min_rows_cols);
  P.zeros(in_n_rows, in_n_rows);

  if (U.n_elem == 0)
    {
    // Nothing to do---leave early.
    L.set_size(in_n_rows, 0);
    U.set_size(0, in_n_cols);
    P.eye(in_n_rows, in_n_rows);
    return true;
    }

  const std::tuple<bool, std::string> result = coot_rt_t::lu(L.get_dev_mem(true),
                                                             U.get_dev_mem(true),
                                                             (U_inplace ? U.get_dev_mem(true) : in.get_dev_mem(true)),
                                                             true,
                                                             P.get_dev_mem(true),
                                                             in_n_rows,
                                                             in_n_cols);
  if (!std::get<0>(result))
    {
    coot_debug_warn_level(3, "lu(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }
