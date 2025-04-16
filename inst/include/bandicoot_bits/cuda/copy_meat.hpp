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
                  const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  cudaError_t error = coot_wrapper(cudaMemcpy2D)(dest,
                                                 sizeof(eT) * n_rows,
                                                 (src.cuda_mem_ptr + src_row_offset + src_col_offset * src_M_n_rows),
                                                 sizeof(eT) * src_M_n_rows,
                                                 sizeof(eT) * n_rows,
                                                 n_cols,
                                                 cudaMemcpyDeviceToHost);

  coot_check_cuda_error(error, "Mat::copy_from_dev_mem(): couldn't access device memory");
  }



template<typename eT>
inline
void
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  cudaError_t error = coot_wrapper(cudaMemcpy)(dest.cuda_mem_ptr, src, N * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(error, "Mat::copy_into_dev_mem(): couldn't access device memory");
  }



/**
 * Use CUDA to copy the source memory to the destination.
 */
template<typename eT>
inline
void
copy_mat(dev_mem_t<eT> dest,
         const dev_mem_t<eT> src,
         const uword n_rows,
         const uword n_cols,
         const uword dest_row_offset,
         const uword dest_col_offset,
         const uword dest_M_n_rows,
         const uword src_row_offset,
         const uword src_col_offset,
         const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  cudaError_t result = coot_wrapper(cudaMemcpy2D)(dest.cuda_mem_ptr + dest_offset,
                                                  sizeof(eT) * dest_M_n_rows,
                                                  src.cuda_mem_ptr + src_offset,
                                                  sizeof(eT) * src_M_n_rows,
                                                  sizeof(eT) * n_rows,
                                                  n_cols,
                                                  cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "coot::cuda::copy_mat(): couldn't copy buffer" );
  }



/*
 * Copy source memory to the destination, changing types.
 */
template<typename eT2, typename eT1>
inline
void
copy_mat(dev_mem_t<eT2> dest,
         const dev_mem_t<eT1> src,
         const uword n_rows,
         const uword n_cols,
         const uword dest_row_offset,
         const uword dest_col_offset,
         const uword dest_M_n_rows,
         const uword src_row_offset,
         const uword src_col_offset,
         const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::convert_type);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_M_n_rows,
      (uword*) &src_M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::copy_mat(): cuLaunchKernel() failed");
  }



/**
 * Copy the source memory to the destination.
 */
template<typename eT2, typename eT1>
inline
void
copy_cube(dev_mem_t<eT2> dest,
          const dev_mem_t<eT1> src,
          // logical size of cube
          const uword n_rows,
          const uword n_cols,
          const uword n_slices,
          // offsets for subviews
          const uword dest_row_offset,
          const uword dest_col_offset,
          const uword dest_slice_offset,
          const uword dest_M_n_rows,
          const uword dest_M_n_cols,
          const uword src_row_offset,
          const uword src_col_offset,
          const uword src_slice_offset,
          const uword src_M_n_rows,
          const uword src_M_n_cols)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::convert_type_cube);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows + dest_slice_offset * dest_M_n_rows * dest_M_n_cols;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows  +  src_slice_offset * src_M_n_rows * src_M_n_cols;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &dest_ptr, // ignored
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_M_n_cols,
      (uword*) &src_M_n_rows, // ignored
      (uword*) &src_M_n_cols, // ignored
      (uword*) &src_M_n_rows,
      (uword*) &src_M_n_cols };

  const kernel_dims dims = three_dimensional_grid_dims(n_rows, n_cols, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::copy_cube(): cuLaunchKernel() failed");
  }
