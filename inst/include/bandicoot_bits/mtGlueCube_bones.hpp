// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename out_eT, typename T1, typename T2, typename mtglue_type>
class mtGlueCube
  : public BaseCube< out_eT, mtGlueCube<out_eT, T1, T2, mtglue_type> >
  {
  public:

  typedef out_eT                                elem_type;
  typedef typename get_pod_type<out_eT>::result pod_type;

  const T1& A;
  const T2& B;
        uword aux_uword;

  inline         ~mtGlueCube();
  inline explicit mtGlueCube(const T1& in_A, const T2& in_B);
  };
