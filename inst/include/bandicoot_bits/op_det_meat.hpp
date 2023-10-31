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
op_det::apply_direct(typename T1::elem_type& out_val, const Base<typename T1::elem_type, T1>& expr)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  // If the input is a diagmat, we can use a specialised implementation.
  if (strip_diagmat<T1>::do_diagmat)
    {
    const strip_diagmat<T1> strip(expr.get_ref());
    const unwrap<typename strip_diagmat<T1>::stored_type> U(strip.M);
    out_val = op_det::apply_diagmat(U.M);
    return true;
    }

  Mat<eT> A(expr.get_ref());

  coot_debug_check( (A.n_rows != A.n_cols), "det(): given matrix must be square sized" );

  if (A.n_rows == 0)
    {
    out_val = (eT) 1;
    return true;
    }

  // Note that Armadillo has specialised variants for extremely small matrices,
  // but here that is too much extra overhead (and the GPU wouldn't really be able
  // to take advantage of it anyway), so we just go full-out with the LU decomposition.
  std::tuple<bool, std::string> result = coot_rt_t::det(A.get_dev_mem(true), A.n_rows, out_val);
  if (std::get<0>(result) != true)
    {
    coot_debug_warn_level(3, "det(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }



template<typename eT>
inline
eT
op_det::apply_diagmat(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  if (X.n_rows == 1 || X.n_cols == 1)
    return coot_rt_t::prod(X.get_dev_mem(false), X.n_elem);
  else
    return 0;
  }
