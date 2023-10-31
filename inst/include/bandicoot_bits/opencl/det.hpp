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
det(dev_mem_t<eT> in, const uword n_rows, eT& out_val)
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

  int* ipiv = cpu_memory::acquire<int>(n_rows);

  if(is_float<eT>::value)
    {
    status = magma_sgetrf_gpu(n_rows, n_rows, in.cl_mem_ptr, 0, n_rows, ipiv, &info);
    }
  else if (is_double<eT>::value)
    {
    status = magma_dgetrf_gpu(n_rows, n_rows, in.cl_mem_ptr, 0, n_rows, ipiv, &info);
    }
  else
    {
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

  // Since we received ipiv on the CPU, we can compute the determinant of P locally.
  eT P_det = 1;
  for (uword i = 0; i < n_rows; ++i)
    {
    if (uword(ipiv[i] - 1) != i)
      {
      P_det *= -1;
      }
    }

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::diag_prod);
  cl_kernel kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::diag_prod_small);
  cl_kernel second_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::prod);
  cl_kernel second_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::prod_small);

  // Force synchronisation of U in case MAGMA is not done yet.
  get_rt().cl_rt.synchronise();

  // Now we can compute the determinant of U by using only the diagonal.
  const eT U_det = generic_reduce<eT, eT>(in, n_rows, "det", kernel, kernel_small, std::make_tuple(/* no extra args */), second_kernel, second_kernel_small, std::make_tuple(/* no extra args */));

  out_val = P_det * U_det;

  return std::make_tuple(true, "");
  }
