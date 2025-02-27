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
 * Compute the trace of the matrix via OpenCL.
 */
template<typename eT>
inline
eT
trace(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::trace(): opencl runtime not valid");

  const uword diag_len = (std::min)(n_rows, n_cols);

  Mat<eT> tmp(1, 1);

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::trace);

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  runtime_t::adapt_uword  cl_n_rows(n_rows);
  runtime_t::adapt_uword          N(diag_len);
  runtime_t::adapt_uword mem_offset(mem.cl_mem_ptr.offset);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),  &(tmp_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),  &(mem.cl_mem_ptr.ptr)    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, mem_offset.size, mem_offset.addr          );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_n_rows.size,  cl_n_rows.addr           );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, N.size,          N.addr                   );

  const size_t global_work_size[1] = { size_t(1) };

  coot_extra_debug_print("clEnqueueNDRangeKernel()");
  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::trace(): couldn't execute kernel" );

  return eT(tmp(0));
  }
