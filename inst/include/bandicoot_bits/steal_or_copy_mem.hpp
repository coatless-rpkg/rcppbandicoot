// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



//
// The steal_or_copy_mem() function is an internal function that,
// if the element types of the two inputs are the same, steals memory.
// If the element types are not the same, the input matrix is converted.
// This is useful for internal operations where out.steal_mem(in) cannot work
// because the element types may differ (and steal_mem() is thus not usable).
//

template<typename out_eT, typename in_eT>
inline
void
steal_or_copy_mem(Mat<out_eT>& out, const Mat<in_eT>& in, const typename enable_if< !is_same_type<out_eT, in_eT>::value >::result* junk = 0)
  {
  coot_ignore(junk);

  out.set_size(in.n_rows, in.n_cols);
  coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                      out.n_rows, out.n_cols,
                      0, 0, out.n_rows,
                      0, 0, in.n_rows);
  }



template<typename eT>
inline
void
steal_or_copy_mem(Mat<eT>& out, Mat<eT>& in)
  {
  out.steal_mem(in);
  }
