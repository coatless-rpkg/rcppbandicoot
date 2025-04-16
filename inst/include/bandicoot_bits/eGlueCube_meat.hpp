// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2017-2025 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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
coot_inline
eGlueCube<T1, T2, eglue_type>::~eGlueCube()
  {
  coot_extra_debug_sigprint();
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
eGlueCube<T1, T2, eglue_type>::eGlueCube(const T1& in_A, const T2& in_B)
  : A(in_A)
  , B(in_B)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size
    (
    A.get_n_rows(), A.get_n_cols(), A.get_n_slices(),
    B.get_n_rows(), B.get_n_cols(), B.get_n_slices(),
    eglue_type::text()
    );
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
uword
eGlueCube<T1, T2, eglue_type>::get_n_rows() const
  {
  return A.get_n_rows();
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
uword
eGlueCube<T1, T2, eglue_type>::get_n_cols() const
  {
  return A.get_n_cols();
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
uword
eGlueCube<T1, T2, eglue_type>::get_n_slices() const
  {
  return A.get_n_slices();
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
uword
eGlueCube<T1, T2, eglue_type>::get_n_elem() const
  {
  return A.get_n_elem();
  }



template<typename T1, typename T2, typename eglue_type>
coot_inline
uword
eGlueCube<T1, T2, eglue_type>::get_n_elem_slice() const
  {
  return A.get_n_elem_slice();
  }
