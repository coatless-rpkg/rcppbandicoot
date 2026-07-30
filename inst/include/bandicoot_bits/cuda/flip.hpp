// Copyright 2026 Andrew Furey (http://andrew.industries/)
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


template <typename eT>
inline
void
fliplr(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols, const uword src_M_n_rows, const uword aux_row1, const uword aux_col1)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
   {
   return;
   }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::fliplr(): CUDA runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::fliplr);

  using ceT = typename cuda_type<eT>::type;
  const ceT* input_mem_ptr = in.cuda_mem_ptr + (aux_col1 * src_M_n_rows + aux_row1);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(input_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &src_M_n_rows, };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void **) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::fliplr(): cuLaunchKernel() failed");
  }


template <typename eT>
inline
void
flipud(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols, const uword src_M_n_rows, const uword aux_row1, const uword aux_col1)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
   {
   return;
   }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "cuda::flipud(): CUDA runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::flipud);

  using ceT = typename cuda_type<eT>::type;
  const ceT* input_mem_ptr = in.cuda_mem_ptr + (aux_col1 * src_M_n_rows + aux_row1);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(input_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &src_M_n_rows, };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void **) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::flipud(): cuLaunchKernel() failed");
  }
