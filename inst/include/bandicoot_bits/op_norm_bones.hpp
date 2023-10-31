// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2022      Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
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



// This class provides a number of methods for directly evaluating vector and matrix norms,
// but we do not implement any optimizations for norm() expressions---so, this class is never
// used as part of a delayed expression; that is, no Op<T1, op_norm> will ever be created.
//
// Whenever norm() is called, it is directly and immediately evaluated.
//
// So, as a result, there are a couple functions not implemented here: apply(), compute_n_rows(), compute_n_cols().
class op_norm
  : public traits_op_default
  {
  public:

  template<typename eT> inline static eT vec_norm_1(const Mat<eT>& X);
  template<typename eT> inline static eT vec_norm_1(const subview<eT>& X);

  template<typename eT> inline static eT vec_norm_2(const Mat<eT>& X);
  template<typename eT> inline static eT vec_norm_2(const subview<eT>& X);

  template<typename eT> inline static eT vec_norm_k(const Mat<eT>& X, const uword p);
  template<typename eT> inline static eT vec_norm_k(const subview<eT>& X, const uword p);

  template<typename eT> inline static eT vec_norm_min(const Mat<eT>& X);
  template<typename eT> inline static eT vec_norm_min(const subview<eT>& X);

  template<typename eT> inline static eT vec_norm_max(const Mat<eT>& X);
  template<typename eT> inline static eT vec_norm_max(const subview<eT>& X);

  template<typename eT> inline static eT mat_norm_1(const Mat<eT>& X);
  template<typename eT> inline static eT mat_norm_1(const subview<eT>& X);

  template<typename eT> inline static eT mat_norm_2(const Mat<eT>& X);
  template<typename eT> inline static eT mat_norm_2(const subview<eT>& X);

  template<typename eT> inline static eT mat_norm_inf(const Mat<eT>& X);
  template<typename eT> inline static eT mat_norm_inf(const subview<eT>& X);
  };
