// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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


/**
 * Given a complex matrix `in_mem`, extract either the real or imaginary
 * elements into `out_mem`.
 */
template<typename eT1, typename eT2>
inline
void
extract_cx(dev_mem_t<eT1> out_mem,
           const uword out_row_offset,
           const uword out_col_offset,
           const uword out_M_n_rows,
           const dev_mem_t<eT2> in_mem,
           const uword in_row_offset,
           const uword in_col_offset,
           const uword in_M_n_rows,
           const uword n_rows,
           const uword n_cols,
           const bool imag)
  {
  coot_extra_debug_sigprint();

  // sanity check
  static_assert( is_cx<eT2>::yes, "eT2 must be complex" );

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::extract_cx(): CUDA runtime not valid");

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT1>(oneway_real_kernel_id::extract_cx);

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  ceT2* in_ptr  =  in_mem.cuda_mem_ptr +  in_row_offset +  in_col_offset *  in_M_n_rows;
  ceT1* out_ptr = out_mem.cuda_mem_ptr + out_row_offset + out_col_offset * out_M_n_rows;

  const uword real_or_imag = (imag) ? 1 : 0;

  void* args[] = {
      (ceT1**) &in_ptr,
      &out_ptr,
      (uword*) &real_or_imag,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &in_M_n_rows,
      (uword*) &out_M_n_rows };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::extract_cx(): cuLaunchKernel() failed");
  }
