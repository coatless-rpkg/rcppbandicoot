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
 * If `lower` is 0, copy the upper triangular part of `in` to the lower triangle and store the resulting symmetric matrix in `out`.
 * If `lower` is 1, copy the lower triangular part of `in` to the upper triangle and store the resulting symmetric matrix in `out`.
 *
 * The input matrix is assumed to be square, with `size` rows and `size` columns.
 */
template<typename eT1, typename eT2>
inline
void
symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::symmat(): OpenCL runtime not valid");

  runtime_t::cq_guard guard;

  cl_int status;

  runtime_t::adapt_uword cl_size(size);

  // If out == in, then we can avoid the copy of the input triangle.
  cl_kernel k;
  if (out.cl_mem_ptr == in.cl_mem_ptr)
    {
    k = (lower == 1) ? get_rt().cl_rt.get_kernel<eT2>(oneway_kernel_id::symmatl_inplace) : get_rt().cl_rt.get_kernel<eT2>(oneway_kernel_id::symmatu_inplace);

    status  = coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem), &(out.cl_mem_ptr));
    status |= coot_wrapper(clSetKernelArg)(k, 1, cl_size.size,   cl_size.addr);
    }
  else
    {
    k = (lower == 1) ? get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::symmatl) : get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::symmatu);

    status  = coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem), &(out.cl_mem_ptr));
    status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem), &(in.cl_mem_ptr));
    status |= coot_wrapper(clSetKernelArg)(k, 2, cl_size.size,   cl_size.addr);
    }

  coot_check_cl_error(status, "coot::opencl::symmat(): could not set arguments for symmat kernel");

  const size_t global_work_size[2] = { size, size };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::symmat(): couldn't execute kernel");
  }
