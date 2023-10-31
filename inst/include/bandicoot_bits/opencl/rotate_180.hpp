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
 * Rotate the given matrix 180 degrees (i.e. flip horizontally and vertically).
 */
template<typename eT>
inline
void
rotate_180(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::rotate_180(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword local_n_rows(n_rows);
  runtime_t::adapt_uword local_n_cols(n_cols);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::rotate_180);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),    &(out.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),    &(in.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, local_n_rows.size, local_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, local_n_cols.size, local_n_cols.addr);

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::rotate_180(): couldn't execute kernel" );
  }
