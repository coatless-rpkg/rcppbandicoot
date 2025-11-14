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
 * Sort the data in each row or column.
 */
template<typename eT>
inline
void
sort(dev_mem_t<eT> mem,
     const uword n_rows,
     const uword n_cols,
     const uword sort_type,
     const uword dim,
     // subview arguments
     const uword row_offset,
     const uword col_offset,
     const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  // If the matrix is empty, don't do anything.
  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k;
  if (dim == 0)
    {
    k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_colwise_ascending : oneway_kernel_id::radix_sort_colwise_descending);
    }
  else
    {
    k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_rowwise_ascending : oneway_kernel_id::radix_sort_rowwise_descending);
    }

  cl_int status = 0;

  const uword mem_offset = mem.cl_mem_ptr.offset + row_offset + col_offset * M_n_rows;

  runtime_t::adapt_uword cl_mem_offset(mem_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_M_n_rows(M_n_rows);

  status  = coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem),     &(mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, cl_mem_offset.size, cl_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(cl_mem),     &(tmp_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 3, cl_n_rows.size,     cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 4, cl_n_cols.size,     cl_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 5, cl_M_n_rows.size,   cl_M_n_rows.addr);

  coot_check_cl_error(status, "coot::opencl::sort(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { (dim == 0) ? n_cols : n_rows };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::sort(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  }



template<typename eT>
inline
void
sort_vec_multiple_workgroups(dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const size_t total_num_threads, const size_t local_group_size)
  {
  coot_extra_debug_sigprint();

  // Since we cannot synchronize across workgroups in OpenCL, we have to do the
  // management of the radix sort process at the CPU level.  The basic strategy
  // is as follows: considering two bits at a time (starting from the LSB), we
  // will:
  //
  //   1. Histogram into counts of the bits [00], [01], [10], [11].
  //   2. Shuffle points into order such that all [00] comes before [01], comes
  //      before [10], comes before [11].
  //
  // The histogramming step is necessary so that we know where to move each
  // point to when we encounter it.  Because we have multiple workgroups,
  // histogramming is a little complex: we must first take a pass to count the
  // number of each bit pattern in each workgroup (organizing the results in a
  // specific way), and then we can shifted-prefix-sum the results, which then
  // gives us the offsets we need to send points to for the shuffle step.

  cl_int status = 0;

  // We already have a cq_guard, so we don't need another.
  cl_kernel k_count = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_multi_wg_bit_count);
  cl_kernel k_shuffle = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_multi_wg_shuffle);

  runtime_t::adapt_uword cl_n_elem(n_elem);

  dev_mem_t<uword> hist;
  dev_mem_t<eT> A_temp;

  hist.cl_mem_ptr      = get_rt().cl_rt.acquire_memory<uword>(4 * total_num_threads);
  A_temp.cl_mem_ptr    = get_rt().cl_rt.acquire_memory<eT>(n_elem);

  dev_mem_t<eT>* A_in = &A;
  dev_mem_t<eT>* A_out = &A_temp;

  for (size_t b = 0; b < 8 * sizeof(eT); b += 2)
    {
    uword step_sort_type = sort_type;
    // If this is the last two bits of a signed integer or floating-point type,
    // we need to sort in a slightly different order to handle the sign bit.
    if (b == (8 * sizeof(eT) - 2))
      {
      if (is_signed<eT>::value && !is_real<eT>::value)
        {
        step_sort_type = 2 + sort_type; // 2 for ascending, 3 for descending
        }
      else if (is_real<eT>::value)
        {
        step_sort_type = 4 + sort_type; // 4 for ascending, 5 for descending
        }
      }

    runtime_t::adapt_uword cl_bit(b);
    runtime_t::adapt_uword cl_sort_type(step_sort_type);
    size_t offset = 0; // TODO: add support for subviews later
    runtime_t::adapt_uword cl_in_offset(A_in->cl_mem_ptr.offset + offset);
    runtime_t::adapt_uword cl_out_offset(A_out->cl_mem_ptr.offset + offset);
    runtime_t::adapt_uword hist_offset(offset);

    // Step 1a: count number of each bit pattern for histogramming.
    status  = coot_wrapper(clSetKernelArg)(k_count, 0, sizeof(cl_mem),    &(A_in->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_count, 1, cl_in_offset.size, cl_in_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 2, sizeof(cl_mem),    &(hist.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_count, 3, hist_offset.size,  hist_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 4, cl_n_elem.size,    cl_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 5, cl_sort_type.size, cl_sort_type.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 6, cl_bit.size,       cl_bit.addr);
    coot_check_cl_error(status, "coot::opencl::sort(): failed to set kernel arguments for bit counting");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_count, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
    coot_check_cl_error(status, "coot::opencl::sort(): failed to run bit counting kernel");

    // Step 1b: prefix-sum per-workgroup histograms into offsets.
    shifted_prefix_sum(hist, 4 * total_num_threads);

    // Step 2: shuffle points using offsets.
    status  = coot_wrapper(clSetKernelArg)(k_shuffle, 0, sizeof(cl_mem),     &(A_in->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 1, cl_in_offset.size,  cl_in_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 2, sizeof(cl_mem),     &(A_out->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 3, cl_out_offset.size, cl_out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 4, sizeof(cl_mem),     &(hist.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 5, cl_n_elem.size,     cl_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 6, cl_sort_type.size,  cl_sort_type.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 7, cl_bit.size,        cl_bit.addr);
    coot_check_cl_error(status, "coot::opencl::sort(): failed to set kernel arguments for shuffle");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_shuffle, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
    coot_check_cl_error(status, "coot::opencl::sort(): failed to run shuffle kernel");

    dev_mem_t<eT>* A_tmp = A_in;
    A_in = A_out;
    A_out = A_tmp;
    }

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(hist.cl_mem_ptr);
  get_rt().cl_rt.release_memory(A_temp.cl_mem_ptr);
  }



template<typename eT>
inline
void
sort_vec(dev_mem_t<eT> A, const uword n_elem, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  // If the vector is empty, don't do anything.
  if (n_elem == 0)
    {
    return;
    }

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_ascending : oneway_kernel_id::radix_sort_descending);

  size_t total_num_threads, local_group_size;
  reduce_kernel_group_info(k, n_elem, "sort", total_num_threads, local_group_size);

  const size_t pow2_num_threads = next_pow2(total_num_threads);

  // If we will have multiple workgroups, we can't use the single-kernel run.
  if (total_num_threads != local_group_size)
    {
    sort_vec_multiple_workgroups(A, n_elem, sort_type, pow2_num_threads, local_group_size);
    return;
    }

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_elem(n_elem);
  runtime_t::adapt_uword cl_mem_offset(A.cl_mem_ptr.offset);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem),                       &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, cl_mem_offset.size,                   cl_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(cl_mem),                       &(tmp_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 3, cl_n_elem.size,                       cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 4, 2 * sizeof(uword) * pow2_num_threads, NULL);
  coot_check_cl_error(status, "coot::opencl::sort(): failed to set kernel arguments");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, 1, NULL, &pow2_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::sort(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  }
