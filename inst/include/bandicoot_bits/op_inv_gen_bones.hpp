// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023-2026 Ryan Curtin (http://www.ratml.org)
// Copyright 2023-2026 Conrad Sanderson (https://conradsanderson.id.au)
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



class op_inv_gen_default
  : public traits_op_passthru
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_inv_gen_default>& in);

  template<typename T1>
  inline static std::tuple<bool, std::string> apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type, T1>& in);
  };



// TODO: implement op_inv_gen_full
