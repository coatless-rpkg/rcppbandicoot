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
  coot_debug_sigprint();

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
  coot_debug_sigprint();

  cudaError_t error = coot_wrapper(cudaMemcpy)(dest.cuda_mem_ptr, src, N * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(error, "Mat::copy_into_dev_mem(): couldn't access device memory");
  }



template<typename T1, typename T2>
inline
void
copy(const Proxy<T1>& out, const Proxy<T2>& in)
  {
  coot_debug_sigprint();

  if (in.is_empty())
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<kernel_id::copy, Proxy<T1>, Proxy<T2>>();

  // this should never happen since coot_rt_t casts T2 to the right dimensions
  coot_static_check( Proxy<T1>::num_dims != Proxy<T2>::num_dims, "coot::cuda::copy(): objects must have the same number of dimensions" );

  const auto& args = construct_args(out, in);
  const kernel_dims dims = grid_dims(out);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args.data(),
      0);

  coot_check_cuda_error( result, "coot::cuda::copy(): cuLaunchKernel() failed" );
  }
