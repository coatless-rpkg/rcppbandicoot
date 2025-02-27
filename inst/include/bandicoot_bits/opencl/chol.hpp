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
 * Compute the Cholesky decomposition using OpenCL.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
chol(dev_mem_t<eT> mem, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  magma_int_t info   = 0;
  magma_int_t status = 0;

  // using MAGMA 2.2

  // OpenCL uses opaque memory pointers which hide the underlying type,
  // so we don't need to do template tricks or casting

  if(is_float<eT>::value)
    {
    status = magma_spotrf_gpu(MagmaUpper, n_rows, mem.cl_mem_ptr.ptr, mem.cl_mem_ptr.offset, n_rows, &info);
    }
  else if(is_double<eT>::value)
    {
    status = magma_dpotrf_gpu(MagmaUpper, n_rows, mem.cl_mem_ptr.ptr, mem.cl_mem_ptr.offset, n_rows, &info);
    }
  else
    {
    return std::make_tuple(false, "not implemented for given type; must be float or double");
    }

  if (status != MAGMA_SUCCESS)
    {
    return std::make_tuple(false, "MAGMA failure in potrf_gpu(): " + magma::error_as_string(status));
    }

  // Process the returned info.
  if (info < 0)
    {
    std::ostringstream oss;
    oss << "parameter " << (-info) << " was incorrect on entry to MAGMA potrf_gpu()";
    return std::make_tuple(false, oss.str());
    }
  else if (info > 0)
    {
    std::ostringstream oss;
    oss << "factorisation failed: the leading minor of order " << info << " is not positive definite";
    return std::make_tuple(false, oss.str());
    }

  // now set the lower triangular part (excluding diagonal) to zero
  cl_int status2 = 0;

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::ltri_set_zero);

  // n_rows == n_cols because the Cholesky decomposition requires square matrices.
  runtime_t::adapt_uword dev_offset(mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword dev_n_rows(n_rows);

  status2 |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(mem.cl_mem_ptr.ptr));
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 1, dev_offset.size, dev_offset.addr);
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 2, dev_n_rows.size, dev_n_rows.addr);
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 3, dev_n_rows.size, dev_n_rows.addr);
  if (status2 != CL_SUCCESS)
    {
    return std::make_tuple(false, "failed to set arguments for kernel ltri_set_zero: " + coot_cl_error::as_string(status2));
    }

  size_t global_work_offset[2] = { 0, 0 };
  size_t global_work_size[2] = { size_t(n_rows), size_t(n_rows) };

  status2 |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);
  if (status2 != CL_SUCCESS)
    {
    return std::make_tuple(false, "failed to run kernel ltri_set_zero: " + coot_cl_error::as_string(status2));
    }

  return std::make_tuple(true, "");
  }
