// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://www.ratml.org)
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



class op_col_as_mat
  : public traits_op_default
  {
  public:

  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const CubeToMatOp<T1, op_col_as_mat>& expr);

  template<typename T1> inline static uword compute_n_rows(const CubeToMatOp<T1, op_col_as_mat>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename T1> inline static uword compute_n_cols(const CubeToMatOp<T1, op_col_as_mat>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  };
