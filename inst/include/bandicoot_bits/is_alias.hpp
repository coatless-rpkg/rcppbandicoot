// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (https://www.ratml.org)
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



template<typename T1, typename T2>
inline
typename enable_if2< is_Mat<T1>::value && is_Mat<T2>::value, bool >::result
is_alias(const T1& A, const T2& B)
  {
  // Shortcut if the objects are the same.
  if (((void*) &A) == ((void*) &B))
    {
    return true;
    }

  return mem_overlaps(A.get_dev_mem(false),
                      0,
                      A.n_elem,
                      B.get_dev_mem(false),
                      0,
                      B.n_elem);
  }



template<typename T1, typename eT2>
inline
typename enable_if2< is_Mat<T1>::value, bool >::result
is_alias(const T1& A, const subview<eT2>& B)
  {
  const uword B_m_n_elem = B.n_rows + B.n_cols * B.m.n_rows;

  return mem_overlaps(A.get_dev_mem(false),
                      0,
                      A.n_elem,
                      B.m.get_dev_mem(false),
                      B.aux_row1 + B.aux_col1 + B.m.n_rows,
                      B_m_n_elem);
  }



template<typename eT1, typename T2>
inline
typename enable_if2< is_Mat<T2>::value, bool >::result
is_alias(const subview<eT1>& A, const T2& B)
  {
  const uword A_m_n_elem = A.n_rows + A.n_cols * A.m.n_rows;

  return mem_overlaps(A.m.get_dev_mem(false),
                      A.aux_row1 + A.aux_col1 * A.m.n_rows,
                      A_m_n_elem,
                      B.get_dev_mem(false),
                      0,
                      B.n_elem);
  }



template<typename eT1, typename eT2>
inline
bool
is_alias(const subview<eT1>& A, const subview<eT2>& B)
  {
  // Shortcut if the objects are the same.
  if (((void*) &A) == ((void*) &B))
    {
    return true;
    }

  const uword A_m_n_elem = A.n_rows + A.n_cols * A.m.n_rows;
  const uword B_m_n_elem = B.n_rows + B.n_cols * B.m.n_rows;

  return mem_overlaps(A.m.get_dev_mem(false),
                      A.aux_row1 + A.aux_col1 * A.m.n_rows,
                      A_m_n_elem,
                      B.m.get_dev_mem(false),
                      B.aux_row1 + B.aux_col1 + B.m.n_rows,
                      B_m_n_elem);
  }
