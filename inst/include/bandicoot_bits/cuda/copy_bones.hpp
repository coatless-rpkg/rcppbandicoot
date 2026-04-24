// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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



template<typename eT>
inline
void
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows);



template<typename eT>
inline
void
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N);



/**
 * Copy the object `in` to `out`.
 */
template<typename T1, typename T2>
inline
void
copy(const Proxy<T1>& out, const Proxy<T2>& in);
