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
 * Find indices of nonzero values in the given vector.
 */
template<typename eT>
inline
void
find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type)
  {
  coot_extra_debug_sigprint();

  // If the vector is empty, don't do anything.
  if (n_elem == 0)
    {
    out_len = 0;
    return;
    }

  runtime_t::cq_guard guard;

  cl_kernel nnz_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::count_nonzeros);

  const size_t num_threads = reduce_kernel_group_size(nnz_k, n_elem, "find");

  // First, allocate temporary memory for the prefix sum.
  dev_mem_t<uword> counts_mem;
  counts_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(num_threads + 1);

  runtime_t::adapt_uword cl_n_elem(n_elem);
  runtime_t::adapt_uword cl_A_offset(A.cl_mem_ptr.offset);

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(nnz_k, 0, sizeof(cl_mem),                    &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(nnz_k, 1, cl_A_offset.size,                  cl_A_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(nnz_k, 2, sizeof(cl_mem),                    &(counts_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(nnz_k, 3, cl_n_elem.size,                    cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(nnz_k, 4, sizeof(uword) * (num_threads + 1), NULL);

  coot_check_cl_error(status, "coot::opencl::find(): could not set kernel arguments for count_nonzeros kernel");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { num_threads };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), nnz_k, k1_work_dim, k1_work_offset, k1_work_size, k1_work_size, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::find(): could not run find_nonzeros kernel");

  get_rt().cl_rt.synchronise();

  const uword total_nonzeros = get_val(counts_mem, num_threads);
  out_len = (k == 0) ? total_nonzeros : (std::min)(k, total_nonzeros);
  out.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(out_len);

  if (out_len == 0)
    {
    // There are no nonzero values---we're done.
    return;
    }

  if (k == 0 || total_nonzeros < k)
    {
    // Get all nonzero elements.
    cl_kernel find_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::find);

    runtime_t::adapt_uword out_offset(out.cl_mem_ptr.offset);

    status  = coot_wrapper(clSetKernelArg)(find_k, 0, sizeof(cl_mem),   &(A.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 1, cl_A_offset.size, cl_A_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 2, sizeof(cl_mem),   &(counts_mem.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 3, sizeof(cl_mem),   &(out.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 4, out_offset.size,  out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 5, cl_n_elem.size,   cl_n_elem.addr);

    coot_check_cl_error(status, "coot::opencl::find(): could not set kernel arguments for find kernel");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), find_k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

    coot_check_cl_error(status, "coot::opencl::find(): could not run find kernel");
    }
  else if (find_type == 0)
    {
    // Get first `k` nonzero elements.
    cl_kernel find_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::find_first);

    runtime_t::adapt_uword cl_k(k);
    runtime_t::adapt_uword out_offset(out.cl_mem_ptr.offset);

    status  = coot_wrapper(clSetKernelArg)(find_k, 0, sizeof(cl_mem),   &(A.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 1, cl_A_offset.size, cl_A_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 2, sizeof(cl_mem),   &(counts_mem.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 3, sizeof(cl_mem),   &(out.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 4, out_offset.size,  out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 5, cl_k.size,        cl_k.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 6, cl_n_elem.size,   cl_n_elem.addr);

    coot_check_cl_error(status, "coot::opencl::find(): could not set kernel arguments for find_first kernel");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), find_k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

    coot_check_cl_error(status, "coot::opencl::find(): could not run find_first kernel");
    }
  else
    {
    // Get last `k` nonzero elements.
    cl_kernel find_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::find_last);

    const uword m = total_nonzeros - k;

    runtime_t::adapt_uword cl_m(m);
    runtime_t::adapt_uword out_offset(out.cl_mem_ptr.offset);

    status  = coot_wrapper(clSetKernelArg)(find_k, 0, sizeof(cl_mem),   &(A.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 1, cl_A_offset.size, cl_A_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 2, sizeof(cl_mem),   &(counts_mem.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 3, sizeof(cl_mem),   &(out.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(find_k, 4, out_offset.size,  out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 5, cl_m.size,        cl_m.addr);
    status |= coot_wrapper(clSetKernelArg)(find_k, 6, cl_n_elem.size,   cl_n_elem.addr);

    coot_check_cl_error(status, "coot::opencl::find(): could not set kernel arguments for find_last kernel");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), find_k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

    coot_check_cl_error(status, "coot::opencl::find(): could not run find_last kernel");
    }

  get_rt().cl_rt.release_memory(counts_mem.cl_mem_ptr);
  }
