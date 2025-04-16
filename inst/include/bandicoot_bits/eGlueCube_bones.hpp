// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2025 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename T1, typename T2, typename eglue_type>
class eGlueCube : public BaseCube< typename T1::elem_type, eGlueCube<T1, T2, eglue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  const SizeProxyCube<T1> A;
  const SizeProxyCube<T2> B;

  coot_inline ~eGlueCube();
  coot_inline  eGlueCube(const T1& in_A, const T2& in_B);

  coot_inline uword get_n_rows() const;
  coot_inline uword get_n_cols() const;
  coot_inline uword get_n_slices() const;
  coot_inline uword get_n_elem() const;
  coot_inline uword get_n_elem_slice() const;
  };
