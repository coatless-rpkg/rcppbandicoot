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
 * Compute the eigendecomposition using OpenCL.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cl_rt.is_valid() == false)
    {
    return std::make_tuple(false, "OpenCL runtime not valid");
    }

  magma_int_t info   = 0;
  magma_int_t status = 0;
  magma_int_t lwork;
  magma_int_t liwork;
  eT* work_mem;
  eT* wA_mem;
  int* iwork_mem;

  magma_vec_t jobz = (eigenvectors) ? MagmaVec : MagmaNoVec;

  // First, compute the workspace size.

  magma_int_t aux_iwork;
  eT aux_work;
  if(is_float<eT>::value)
    {
    // Workspace size query.
    status = magma_ssyevd_gpu(jobz, MagmaUpper, n_rows, NULL, 0, n_rows, NULL, NULL, n_rows, (float*) &aux_work, -1, &aux_iwork, -1, &info);
    }
  else if (is_double<eT>::value)
    {
    status = magma_dsyevd_gpu(jobz, MagmaUpper, n_rows, NULL, 0, n_rows, NULL, NULL, n_rows, (double*) &aux_work, -1, &aux_iwork, -1, &info);
    }
  else
    {
    return std::make_tuple(false, "not implemented for given type: must be float or double");
    }

  if (status != MAGMA_SUCCESS)
    {
    if (info < 0)
      {
      std::ostringstream oss;
      oss << "parameter " << (-info) << " was incorrect on entry to MAGMA syevd_gpu() workspace size query";
      return std::make_tuple(false, oss.str());
      }
    else
      {
      return std::make_tuple(false, "MAGMA failure in syevd_gpu() workspace size query: " + magma::error_as_string(status));
      }
    }

  // Get workspace sizes and allocate.
  lwork = (magma_int_t) aux_work;
  liwork = aux_iwork;

  eT* eigenvalues_cpu = cpu_memory::acquire<eT>(n_rows);
  work_mem = cpu_memory::acquire<eT>(lwork);
  wA_mem = cpu_memory::acquire<eT>(n_rows * n_rows);
  iwork_mem = cpu_memory::acquire<int>(liwork);

  if (is_float<eT>::value)
    {
    status = magma_ssyevd_gpu(jobz, MagmaUpper, n_rows, mem.cl_mem_ptr.ptr, mem.cl_mem_ptr.offset, n_rows, (float*) eigenvalues_cpu, (float*) wA_mem, n_rows, (float*) work_mem, lwork, iwork_mem, liwork, &info);
    }
  else if(is_double<eT>::value)
    {
    status = magma_dsyevd_gpu(jobz, MagmaUpper, n_rows, mem.cl_mem_ptr.ptr, mem.cl_mem_ptr.offset, n_rows, (double*) eigenvalues_cpu, (double*) wA_mem, n_rows, (double*) work_mem, lwork, iwork_mem, liwork, &info);
    }
  else
    {
    cpu_memory::release(eigenvalues_cpu);
    cpu_memory::release(work_mem);
    cpu_memory::release(wA_mem);
    cpu_memory::release(iwork_mem);
    return std::make_tuple(false, "not implemented for given type; must be float or double");
    }

  // Process the returned info.
  if (status != MAGMA_SUCCESS)
    {
    cpu_memory::release(eigenvalues_cpu);
    cpu_memory::release(work_mem);
    cpu_memory::release(wA_mem);
    cpu_memory::release(iwork_mem);

    if (info < 0)
      {
      std::ostringstream oss;
      oss << "parameter " << (-info) << " was incorrect on entry to MAGMA syevd_gpu()";
      return std::make_tuple(false, oss.str());
      }
    else if (info > 0)
      {
      std::ostringstream oss;
      if (eigenvectors)
        {
        oss << "eigendecomposition failed: could not compute an eigenvalue while working on block submatrix " << info;
        }
      else
        {
        oss << "eigendecomposition failed: " << info << " off-diagonal elements of an intermediate tridiagonal form did not converge to 0";
        }
      return std::make_tuple(false, oss.str());
      }
    }

  // Copy eigenvalues to the device memory we were given.
  copy_into_dev_mem<eT>(eigenvalues, eigenvalues_cpu, n_rows);

  cpu_memory::release(eigenvalues_cpu);
  cpu_memory::release(work_mem);
  cpu_memory::release(wA_mem);
  cpu_memory::release(iwork_mem);

  return std::make_tuple(true, "");
  }
