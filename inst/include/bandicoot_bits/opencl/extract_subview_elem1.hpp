// Copyright 2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT1, typename eT2>
inline
void
extract_subview_elem1(dev_mem_t<eT2> out_mem,
                      const dev_mem_t<eT1> in_mem,
                      const dev_mem_t<uword> in_locs,
                      const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "opencl::extract_subview_elem1(): OpenCL runtime not valid");

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::extract_sve1);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_out_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_in_offset(in_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_in_locs_offset(in_locs.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_n_elem(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),         &out_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_out_offset.size,     cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),         &in_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_in_offset.size,      cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),         &in_locs.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_in_locs_offset.size, cl_in_locs_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_n_elem.size,         cl_n_elem.addr);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::extract_subview_elem1(): couldn't set kernel arguments");

  const size_t work_size = size_t(n_elem);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::extract_subview_elem1(): couldn't execute kernel");
  }
