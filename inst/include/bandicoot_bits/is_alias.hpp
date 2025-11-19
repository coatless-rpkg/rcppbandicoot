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



// This unspecialized version is used for matrices, rows, columns, and cubes.
template<typename T1>
struct alias_details
  {
  static coot_inline dev_mem_t<typename T1::elem_type> get_dev_mem(const T1& M) { return M.get_dev_mem(false); }
  static constexpr   uword                             get_offset(const T1&)    { return 0; }
  static coot_inline uword                             get_n_elem(const T1& M)  { return M.n_elem; }
  };



template<typename eT>
struct alias_details< subview<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const subview<eT>& M) { return M.m.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const subview<eT>& M)  { return M.aux_row1 + M.aux_col1 * M.m.n_rows; }
  static coot_inline uword                             get_n_elem(const subview<eT>& M)  { return M.n_rows + M.n_cols * M.m.n_rows; }
  };



template<typename eT>
struct alias_details< subview_col<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const subview_col<eT>& M) { return M.m.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const subview_col<eT>& M)  { return M.aux_row1 + M.aux_col1 * M.m.n_rows; }
  static coot_inline uword                             get_n_elem(const subview_col<eT>& M)  { return M.n_rows + M.n_cols * M.m.n_rows; }
  };



template<typename eT>
struct alias_details< subview_row<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const subview_row<eT>& M) { return M.m.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const subview_row<eT>& M)  { return M.aux_row1 + M.aux_col1 * M.m.n_rows; }
  static coot_inline uword                             get_n_elem(const subview_row<eT>& M)  { return M.n_rows + M.n_cols * M.m.n_rows; }
  };



template<typename eT>
struct alias_details< diagview<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const diagview<eT>& M) { return M.m.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const diagview<eT>& M)  { return M.mem_offset; }
  static coot_inline uword                             get_n_elem(const diagview<eT>& M)  { return M.n_rows * (M.m.n_rows + 1); }
  };



template<typename eT>
struct alias_details< Cube<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const Cube<eT>& M) { return M.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const Cube<eT>&)    { return 0; }
  static coot_inline uword                             get_n_elem(const Cube<eT>& M)  { return M.n_elem; }
  };



template<typename eT>
struct alias_details< subview_cube<eT> >
  {
  static coot_inline dev_mem_t<eT>                     get_dev_mem(const subview_cube<eT>& M) { return M.m.get_dev_mem(false); }
  static coot_inline uword                             get_offset(const subview_cube<eT>& M)  { return M.aux_row1 + M.aux_col1 * M.m.n_rows + M.aux_slice1 * M.m.n_rows * M.m.n_cols; }
  static coot_inline uword                             get_n_elem(const subview_cube<eT>& M)  { return M.n_rows + M.n_cols * M.m.n_rows + M.n_slices * M.m.n_rows * M.m.n_cols; }
  };



template<typename T1, typename T2>
inline
bool
is_alias(const T1& A, const T2& B)
  {
  return mem_overlaps(alias_details<T1>::get_dev_mem(A),
                      alias_details<T1>::get_offset(A),
                      alias_details<T1>::get_n_elem(A),
                      alias_details<T2>::get_dev_mem(B),
                      alias_details<T2>::get_offset(B),
                      alias_details<T2>::get_n_elem(B));
  }
