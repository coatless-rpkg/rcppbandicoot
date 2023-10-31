// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



class op_det
  : public traits_op_default
  {
  public:

  template<typename T1>
  inline static bool apply_direct(typename T1::elem_type& out_val, const Base<typename T1::elem_type, T1>& expr);

  template<typename eT>
  inline static eT apply_diagmat(const Mat<eT>& X);
  };
