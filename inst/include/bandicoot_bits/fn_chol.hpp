// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017-2023 Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename T1>
inline
bool
chol(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();

  out = X.get_ref();

  coot_debug_check( out.n_rows != out.n_cols, "chol(): given matrix must be square sized" );

  if (out.n_rows == 0 || out.n_cols == 0)
    {
    return true; // nothing to do, matrix is empty
    }

  std::tuple<bool, std::string> result = coot_rt_t::chol(out.get_dev_mem(true), out.n_rows);
  if (std::get<0>(result) == false)
    {
    out.reset();
    coot_debug_warn_level(3, "coot::chol(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }



template<typename T1>
coot_warn_unused
inline
Mat<typename T1::elem_type>
chol(const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();

  Mat<typename T1::elem_type> out(X.get_ref());

  coot_debug_check( out.n_rows != out.n_cols, "chol(): given matrix must be square sized" );

  if (out.n_rows == 0 || out.n_cols == 0)
    {
    return out; // nothing to do, matrix is empty
    }

  std::tuple<bool, std::string> result = coot_rt_t::chol(out.get_dev_mem(true), out.n_rows);
  if (std::get<0>(result) == false)
    {
    coot_stop_runtime_error("coot::chol(): " + std::get<1>(result));
    }

  return out;
  }
