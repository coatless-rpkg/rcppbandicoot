// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
 * Generate a vector with num elements; the values of the elements are linearly spaced from start to (and including) end via OpenCL.
 */
template<typename eT>
inline
void
linspace(dev_mem_t<eT> mem, const uword mem_incr, const eT start, const eT end, const uword num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "opencl::linspace(): OpenCL runtime not valid");

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::linspace);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_mem_incr(mem_incr);
  runtime_t::adapt_uword N(num);
  runtime_t::adapt_uword dev_offset(mem.cl_mem_ptr.offset);

  cl_int status = 0;

  const eT step = static_cast<eT>(end - start) / (num - 1);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),   &mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, dev_offset.size,  dev_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, cl_mem_incr.size, cl_mem_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(eT),       &start);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(eT),       &end);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(eT),       &step);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, N.size,           N.addr);

  size_t work_size = size_t(num);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::linspace(): couldn't execute kernel");
  }
