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
 * Compute the 3-dimensional cross-product of A and B into out.
 */
template<typename eT1, typename eT2>
inline
void
cross(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const dev_mem_t<eT1> B)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::cross(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::cross);

  cl_int status = 0;

  runtime_t::adapt_uword out_offset(out.cl_mem_ptr.offset);
  runtime_t::adapt_uword A_offset(A.cl_mem_ptr.offset);
  runtime_t::adapt_uword B_offset(B.cl_mem_ptr.offset);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),  &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, out_offset.size, out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),  &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, A_offset.size,   A_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),  &(B.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, B_offset.size,   B_offset.addr);

  const size_t global_work_size = 3;

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::cross(): couldn't execute kernel" );
  }
