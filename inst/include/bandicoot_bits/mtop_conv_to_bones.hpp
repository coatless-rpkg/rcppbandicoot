// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (http://www.ratml.org
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



class mtop_conv_to
  : public traits_op_passthru
  {
  public:

  template<typename out_eT, typename T1>
  static inline void apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X);

  template<typename out_eT, typename T1>
  static inline void apply(Cube<out_eT>& out, const mtOpCube<out_eT, T1, mtop_conv_to>& X);

  // Compute the sizes of the output.
  template<typename out_eT, typename T1> static inline uword compute_n_rows(const mtOp<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols);
  template<typename out_eT, typename T1> static inline uword compute_n_cols(const mtOp<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols);

  template<typename out_eT, typename T1> static inline uword compute_n_rows(  const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename out_eT, typename T1> static inline uword compute_n_cols(  const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  template<typename out_eT, typename T1> static inline uword compute_n_slices(const mtOpCube<out_eT, T1, mtop_conv_to>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  };
