// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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



class mtop_real
  : public traits_op_default
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_real>& in);

  template<typename out_eT, typename T1> inline static uword compute_n_rows(const mtOp<out_eT, T1, mtop_real>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename out_eT, typename T1> inline static uword compute_n_cols(const mtOp<out_eT, T1, mtop_real>& op, const uword in_n_rows, const uword in_n_cols);
  };
