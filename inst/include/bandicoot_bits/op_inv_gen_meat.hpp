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



template<typename T1>
inline
void
op_inv_gen_default::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_inv_gen_default>& in)
  {
  coot_debug_sigprint();
  
  const std::tuple<bool, std::string> status = apply_direct(out, in.m);
  
  if(std::get<0>(status) == false)
    {
    out.reset();
    coot_stop_runtime_error("inv(): " + std::get<1>(status));
    }
  }



template<typename T1>
inline
std::tuple<bool, std::string>
op_inv_gen_default::apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type, T1>& in)
  {
  coot_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  // better-than-nothing implementation until magma_?getri_gpu() is ported
  
  Mat<eT> A(in.get_ref());  // make a copy, as matrix 'A' will be destroyed by coot_rt_t::solve_square_fast()
  
  coot_conform_check( (A.is_square() == false), "inv(): given matrix must be square sized" );
  
  if(A.is_empty())
    {
    out.set_size(0,0);
    return std::make_tuple(true, std::string(""));
    }
  
  out.eye(A.n_rows, A.n_rows);  // coot_rt_t::solve_square_fast() will overwrite matrix 'out' with the inverse
  
  return coot_rt_t::solve_square_fast(A.get_dev_mem(true), false, out.get_dev_mem(true), out.n_rows, out.n_cols);
  }
