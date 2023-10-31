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



// This "special" unwrapper for the cor() operation checks to see if the
// unwrapped input a row vector, and if so, treats it as a column vector
// instead.  This requires a little extra handling in case the object is a
// subview.
template<typename T1>
struct special_cor_cov_unwrap : public no_conv_unwrap<T1>
  {
  inline special_cor_cov_unwrap(const T1& t)
    : no_conv_unwrap<T1>(t)
    , use_local_mat(false)
    {
    // If the input is a row vector, we treat it as a column vector instead.
    if (no_conv_unwrap<T1>::M.n_rows == 1)
      {
      access::rw(local_mat) = unwrap_cor_trans(no_conv_unwrap<T1>::M);
      access::rw(use_local_mat) = true;
      }
    }

  typedef typename no_conv_unwrap<T1>::stored_type::elem_type eT;

  const uword         get_n_rows()                 const { return (use_local_mat) ? local_mat.n_rows            : no_conv_unwrap<T1>::M.n_rows;          }
  const uword         get_n_cols()                 const { return (use_local_mat) ? local_mat.n_cols            : no_conv_unwrap<T1>::M.n_cols;          }
  const dev_mem_t<eT> get_dev_mem(const bool sync) const { return (use_local_mat) ? local_mat.get_dev_mem(sync) : no_conv_unwrap<T1>::get_dev_mem(sync); }
  const uword         get_row_offset()             const { return (use_local_mat) ? 0                           : no_conv_unwrap<T1>::get_row_offset();  }
  const uword         get_col_offset()             const { return (use_local_mat) ? 0                           : no_conv_unwrap<T1>::get_col_offset();  }
  const uword         get_M_n_rows()               const { return (use_local_mat) ? local_mat.n_rows            : no_conv_unwrap<T1>::get_M_n_rows();    }

  const bool use_local_mat;
  const Mat<eT> local_mat;



  // helper functions
  template<typename eT>
  inline
  Mat<eT>
  unwrap_cor_trans(const Mat<eT>& in)
    {
    // Here we can just make an alias.
    return Mat<eT>(in.get_dev_mem(false), in.n_cols, in.n_rows);
    }



  template<typename eT>
  inline
  Mat<eT>
  unwrap_cor_trans(const subview<eT>& in)
    {
    // For transposing a subview, we have to extract it.
    return Mat<eT>(in.t());
    }



  };
