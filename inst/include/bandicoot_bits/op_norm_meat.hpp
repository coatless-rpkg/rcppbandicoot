// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2022      Marcus Edel (http://kurg.org)
// Copyright 2008-2023 Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename eT>
inline
eT
op_norm::vec_norm_1(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_1(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_1(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return vec_norm_1(tmp);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_2(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_2(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_2(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return vec_norm_2(tmp);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_k(const Mat<eT>& X, const uword k)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_k(X.get_dev_mem(false), X.n_elem, k);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_k(const subview<eT>& X, const uword k)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return vec_norm_k(tmp, k);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_min(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_min(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_min(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return vec_norm_min(tmp);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_max(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::max_abs(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_max(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return vec_norm_max(tmp);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_1(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  // TODO: a dedicated implementation could be faster
  Mat<eT> result = max( sum( abs(X), 0 ), 1 );
  return eT(result[0]);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_1(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return mat_norm_1(tmp);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_2(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  // TODO: once is_finite() is implemented, handle this warning
//  if (X.is_finite() == false)
//    {
//    coot_debug_warn_level(1, "norm(): given matrix has non-finite elements");
//    }

  Col<eT> s;
  svd(s, X);

  return (s.n_elem > 0) ? s[0] : eT(0);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_2(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return mat_norm_2(tmp);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_inf(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  // TODO: a dedicated implementation could be faster
  Mat<eT> result = max( sum( abs(X), 1 ), 0 );
  return eT(result[0]);
  }



template<typename eT>
inline
eT
op_norm::mat_norm_inf(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: a better implementation would be much more efficient and avoid this
  // extraction.
  Mat<eT> tmp(X);
  return mat_norm_inf(tmp);
  }
