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



/**
 * Compute the trace of the matrix via CUDA.
 */
template<typename eT>
inline
eT
trace(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::trace(): cuda runtime not valid");

  const uword diag_len = (std::min)(n_rows, n_cols);

  Mat<eT> tmp(1, 1);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::trace);

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(mem.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &diag_len };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      // TODO: better trace() kernel
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::trace(): cuLaunchKernel() failed");

  return eT(tmp(0));
  }
