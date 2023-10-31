// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2022 Gopi Tatiraju
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



class glue_join_cols
  : public traits_glue_default
  {
  public:

  // Note that it's okay if T1 and T2 contain conversions.
  template<typename out_eT, typename T1, typename T2> static inline void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_join_cols>& glue);

  // Higher-arity variants, called directly
  template<typename eT, typename T1, typename T2, typename T3>              static inline void apply(Mat<eT>& out, const T1& A, const T2& B, const T3& C, const std::string& func_name);
  template<typename eT, typename T1, typename T2, typename T3, typename T4> static inline void apply(Mat<eT>& out, const T1& A, const T2& B, const T3& C, const T4& D, const std::string& func_name);

  template<typename T1, typename T2> static inline uword compute_n_rows(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> static inline uword compute_n_cols(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };
