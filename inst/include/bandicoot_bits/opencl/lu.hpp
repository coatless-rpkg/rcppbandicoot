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
 * Compute the LU factorisation using OpenCL.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  // We'll perform the operation in-place in `in`.
  // If n_rows <= n_cols, then `in` can safely be the same memory as `U`.

  magma_int_t info   = 0;
  magma_int_t status = 0; // NOTE: all paths through dgetrf and sgetrf just return status == info...

  const uword ipiv_size = (std::min)(n_rows, n_cols);
  int* ipiv = cpu_memory::acquire<int>(ipiv_size);

  if(is_float<eT>::value)
    {
    status = magma_sgetrf_gpu(n_rows, n_cols, in.cl_mem_ptr, 0, n_rows, ipiv, &info);
    }
  else if (is_double<eT>::value)
    {
    status = magma_dgetrf_gpu(n_rows, n_cols, in.cl_mem_ptr, 0, n_rows, ipiv, &info);
    }
  else
    {
    cpu_memory::release(ipiv);
    return std::make_tuple(false, "unknown data type, must be float or double");
    }

  if (status != MAGMA_SUCCESS)
    {
    cpu_memory::release(ipiv);
    if (info < 0)
      {
      std::ostringstream oss;
      oss << "parameter " << -info << " was incorrect in call to MAGMA getrf_gpu()";
      return std::make_tuple(false, oss.str());
      }
    else
      {
      std::ostringstream oss;
      oss << "decomposition failed, U(" << (info - 1) << ", " << (info - 1) << ") was found to be 0";
      return std::make_tuple(false, oss.str());
      }
    }

  // First the pivoting needs to be "unwound" into a way where we can make P.
  uword* ipiv2 = cpu_memory::acquire<uword>(n_rows);
  for (uword i = 0; i < n_rows; ++i)
    {
    ipiv2[i] = i;
    }

  for (uword i = 0; i < ipiv_size; ++i)
    {
    const uword k = (uword) ipiv[i] - 1; // the original data is returned in a 1-indexed way

    if (ipiv2[i] != ipiv2[k])
      {
      std::swap( ipiv2[i], ipiv2[k] );
      }
    }

  dev_mem_t<uword> ipiv_gpu;
  ipiv_gpu.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(n_rows);
  copy_into_dev_mem(ipiv_gpu, ipiv2, n_rows);
  cpu_memory::release(ipiv);
  cpu_memory::release(ipiv2);

  // Now extract the lower triangular part (excluding diagonal).  This is done with a custom kernel.
  cl_int status2 = 0;

  runtime_t::cq_guard guard;

  // If pivoting is allowed, we extract L as-is.  Otherwise, we apply the pivoting to L while we extract it.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(pivoting ? oneway_real_kernel_id::lu_extract_l : oneway_real_kernel_id::lu_extract_pivoted_l);

  runtime_t::adapt_uword dev_n_rows(n_rows);
  runtime_t::adapt_uword dev_n_cols(n_cols);

  status2  = coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(L.cl_mem_ptr));
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), &(U.cl_mem_ptr));
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem), &(in.cl_mem_ptr));
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 3, dev_n_rows.size, dev_n_rows.addr);
  status2 |= coot_wrapper(clSetKernelArg)(kernel, 4, dev_n_cols.size, dev_n_cols.addr);
  if (!pivoting)
    {
    status2 |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(cl_mem), &(ipiv_gpu.cl_mem_ptr));
    }

  if (status2 != CL_SUCCESS)
    {
    get_rt().cl_rt.release_memory(ipiv_gpu.cl_mem_ptr);
    return std::make_tuple(false, "failed to set arguments for kernel " + (pivoting ? std::string("lu_extract_l") : std::string("lu_extract_pivoted_l")));
    }

  size_t global_work_offset[2] = { 0, 0 };
  size_t global_work_size[2] = { size_t(n_rows), size_t(std::max(n_rows, n_cols)) };

  status2 = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);

  if (status2 != CL_SUCCESS)
    {
    get_rt().cl_rt.release_memory(ipiv_gpu.cl_mem_ptr);
    return std::make_tuple(false, "failed to run kernel " + (pivoting ? std::string("lu_extract_l") : std::string("lu_extract_pivoted_l")));
    }

  // If pivoting was allowed, extract the permutation matrix.
  if (pivoting)
    {
    kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::lu_extract_p);

    status2  = coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),    &(P.cl_mem_ptr));
    status2 |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),    &(ipiv_gpu.cl_mem_ptr));
    status2 |= coot_wrapper(clSetKernelArg)(kernel, 2, dev_n_rows.size,   dev_n_rows.addr);

    if (status2 != CL_SUCCESS)
      {
      get_rt().cl_rt.release_memory(ipiv_gpu.cl_mem_ptr);
      return std::make_tuple(false, "failed to set arguments for kernel lu_extract_p");
      }

    size_t global_work_offset_2 = 0;
    size_t global_work_size_2   = n_rows;

    status2 = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, &global_work_offset_2, &global_work_size_2, NULL, 0, NULL, NULL);

    if (status2 != CL_SUCCESS)
      {
      get_rt().cl_rt.release_memory(ipiv_gpu.cl_mem_ptr);
      return std::make_tuple(false, "failed to run kernel lu_extract_p");
      }
    }

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(ipiv_gpu.cl_mem_ptr);

  return std::make_tuple(true, "");
  }
