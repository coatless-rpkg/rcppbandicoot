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



template<typename eT>
inline
std::tuple<bool, std::string>
svd(dev_mem_t<eT> U,
    dev_mem_t<eT> S,
    dev_mem_t<eT> V,
    dev_mem_t<eT> A,
    const uword n_rows,
    const uword n_cols,
    const bool compute_u_vt)
  {
  coot_stop_runtime_error("opencl::svd(): unsupported type");

  return std::make_tuple(false, "unsupported type");
  }



/**
 * Compute the singular value decomposition using CUDA (cuSolverDn).
 *
 * This expects that n_rows < n_cols; if that's not true, transpose A (and handle the slightly different results).
 *
 * Note that this function will not throw but instead will return a bool indicating success or failure.
 */
inline
std::tuple<bool, std::string>
svd(dev_mem_t<float> U,
    dev_mem_t<float> S,
    dev_mem_t<float> V,
    dev_mem_t<float> A,
    const uword n_rows,
    const uword n_cols,
    const bool compute_u_vt)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  if (n_rows < n_cols)
    {
    return std::make_tuple(false, "n_rows must be greater than or equal to n_cols");
    }

  magma_vec_t job = compute_u_vt ? MagmaAllVec : MagmaNoVec;

  // First, compute the size of the workspace that we need,
  magma_int_t lwork = -1;
  float tmp;
  magma_int_t info;
  magma_sgesvd(job, job, n_rows, n_cols, NULL, n_rows, NULL, NULL, n_rows, NULL, n_cols, &tmp, lwork, &info);
  if (info != 0)
    {
    return std::make_tuple(false, "magma_sgesvd() workspace call failed with error " + magma::error_as_string(info));
    }
  lwork = magma_int_t(tmp);

  // magma_sgesvd() actually requires data to be on the CPU to start its work,
  // since it is a hybrid algorithm.  Therefore, we'll collect the A matrix from
  // the GPU.

  // Now, allocate space for the workspace.
  size_t U_size = compute_u_vt ? n_rows * n_rows : 0;
  size_t VT_size = compute_u_vt ? n_cols * n_cols : 0;
  float* cpu_mem = cpu_memory::acquire<float>((n_rows * n_cols) + // A
                                              U_size + // U
                                              std::min(n_rows, n_cols) + // s
                                              VT_size + // VT
                                              lwork);
  float* cpu_A = cpu_mem;
  float* cpu_U = cpu_A + n_rows * n_cols;
  float* cpu_s = cpu_U + U_size;
  float* cpu_VT = cpu_s + std::min(n_rows, n_cols);
  float* cpu_work = cpu_VT + VT_size;

  copy_from_dev_mem(cpu_A, A, n_rows, n_cols, 0, 0, n_rows);

  // Now actually compute the SVD.
  magma_sgesvd(job,
               job,
               n_rows,
               n_cols,
               cpu_A,
               n_rows,
               cpu_s,
               cpu_U,
               n_rows,
               cpu_VT,
               n_cols,
               cpu_work,
               lwork,
               &info);
  if (info != 0)
    {
    cpu_memory::release(cpu_mem);
    return std::make_tuple(false, "magma_sgesvd() call failed with error " + magma::error_as_string(info));
    }

  // Copy the results back onto the GPU.
  // TODO: could make these calls asynchronous
  if (compute_u_vt)
    {
    copy_into_dev_mem(U, cpu_U, n_rows * n_rows);
    copy_into_dev_mem(V, cpu_VT, n_cols * n_cols);
    }
  copy_into_dev_mem(S, cpu_s, std::min(n_rows, n_cols));

  cpu_memory::release(cpu_mem);
  return std::make_tuple(true, "");
  }



inline
std::tuple<bool, std::string>
svd(dev_mem_t<double> U,
    dev_mem_t<double> S,
    dev_mem_t<double> V,
    dev_mem_t<double> A,
    const uword n_rows,
    const uword n_cols,
    const bool compute_u_vt)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  if (n_rows < n_cols)
    {
    return std::make_tuple(false, "n_rows must be greater than or equal to n_cols");
    }

  magma_vec_t job = compute_u_vt ? MagmaAllVec : MagmaNoVec;

  // First, compute the size of the workspace that we need,
  magma_int_t lwork = -1;
  double tmp;
  magma_int_t info;
  magma_dgesvd(job, job, n_rows, n_cols, NULL, n_rows, NULL, NULL, n_rows, NULL, n_cols, &tmp, lwork, &info);
  if (info != 0)
    {
    return std::make_tuple(false, "magma_dgesvd() workspace call failed with error " + magma::error_as_string(info));
    }
  lwork = magma_int_t(tmp);

  // magma_dgesvd() actually requires data to be on the CPU to start its work,
  // since it is a hybrid algorithm.  Therefore, we'll collect the A matrix from
  // the GPU.

  // Now, allocate space for the workspace.
  size_t U_size = compute_u_vt ? n_rows * n_rows : 0;
  size_t VT_size = compute_u_vt ? n_cols * n_cols : 0;
  double* cpu_mem = cpu_memory::acquire<double>((n_rows * n_cols) + // A
                                                U_size + // U
                                                std::min(n_rows, n_cols) + // s
                                                VT_size + // VT
                                                lwork);
  double* cpu_A = cpu_mem;
  double* cpu_U = cpu_A + n_rows * n_cols;
  double* cpu_s = cpu_U + U_size;
  double* cpu_VT = cpu_s + std::min(n_rows, n_cols);
  double* cpu_work = cpu_VT + VT_size;

  copy_from_dev_mem(cpu_A, A, n_rows, n_cols, 0, 0, n_rows);

  // Now actually compute the SVD.
  magma_dgesvd(job,
               job,
               n_rows,
               n_cols,
               cpu_A,
               n_rows,
               cpu_s,
               cpu_U,
               n_rows,
               cpu_VT,
               n_cols,
               cpu_work,
               lwork,
               &info);
  if (info != 0)
    {
    cpu_memory::release(cpu_mem);
    return std::make_tuple(false, "magma_dgesvd() call failed with error " + magma::error_as_string(info));
    }

  // Copy the results back onto the GPU.
  // TODO: could make these calls asynchronous
  if (compute_u_vt)
    {
    copy_into_dev_mem(U, cpu_U, n_rows * n_rows);
    copy_into_dev_mem(V, cpu_VT, n_cols * n_cols);
    }
  copy_into_dev_mem(S, cpu_s, std::min(n_rows, n_cols));

  cpu_memory::release(cpu_mem);
  return std::make_tuple(true, "");
  }
