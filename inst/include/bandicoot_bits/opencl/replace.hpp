// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
 * Replace `val_find` with `val_replace`.
 */
template<typename eT>
inline
void
replace(dev_mem_t<eT> mem, const uword n_elem, const eT val_find, const eT val_replace)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::replace);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &mem.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(eT),     &val_find      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT),     &val_replace   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, N.size,         N.addr         );

  size_t work_size = size_t(n_elem);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::replace(): couldn't execute kernel");
  }
