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
 * Solve the system A * X = B or A^T * X = B for square A using the LU factorisation using OpenCL.
 *
 * A is of size n_rows x n_rows, and is destroyed.
 * B is of size n_rows x n_cols, and is replaced with X
 */
template<typename eT>
inline
std::tuple<bool, std::string>
solve_square_fast(dev_mem_t<eT> A, const bool trans_A, dev_mem_t<eT> B, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  // First perform the LU decomposition of A in-place.
  magma_int_t info   = 0;
  magma_int_t status = 0; // NOTE: all paths through dgetrf and sgetrf just return status == info...

  int* ipiv = cpu_memory::acquire<int>(n_rows);

  if (is_float<eT>::value)
    {
    status = magma_sgetrf_gpu(n_rows,
                              n_rows,
                              A.cl_mem_ptr.ptr,
                              A.cl_mem_ptr.offset,
                              n_rows,
                              ipiv,
                              &info);
    }
  else if (is_double<eT>::value)
    {
    status = magma_dgetrf_gpu(n_rows,
                              n_rows,
                              A.cl_mem_ptr.ptr,
                              A.cl_mem_ptr.offset,
                              n_rows,
                              ipiv,
                              &info);
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

  // Now use the LU-decomposed A to solve the system.

  if(is_float<eT>::value)
    {
    status = magma_sgetrs_gpu((trans_A ? MagmaTrans : MagmaNoTrans),
                              n_rows,
                              n_cols,
                              A.cl_mem_ptr.ptr,
                              A.cl_mem_ptr.offset,
                              n_rows,
                              ipiv,
                              B.cl_mem_ptr.ptr,
                              B.cl_mem_ptr.offset,
                              n_rows,
                              &info);
    }
  else if (is_double<eT>::value)
    {
    status = magma_dgetrs_gpu((trans_A ? MagmaTrans : MagmaNoTrans),
                              n_rows,
                              n_cols,
                              A.cl_mem_ptr.ptr,
                              A.cl_mem_ptr.offset,
                              n_rows,
                              ipiv,
                              B.cl_mem_ptr.ptr,
                              B.cl_mem_ptr.offset,
                              n_rows,
                              &info);
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
      oss << "parameter " << -info << " was incorrect in call to MAGMA getrs_gpu()";
      return std::make_tuple(false, oss.str());
      }
    else
      {
      return std::make_tuple(false, "call to getrs_gpu() failed");
      }
    }

  return std::make_tuple(true, "");
  }
