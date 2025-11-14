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
void
stable_sort_index_vec_multiple_workgroups(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const size_t total_num_threads, const size_t local_group_size)
  {
  coot_extra_debug_sigprint();

  // For a stable sort, we can never shuffle points in reverse order,
  // so we have to take a different strategy for floating-point numbers
  // (which normally unpack in reverse order for the last two bits).
  // Non-floating-point numbers will use the regular
  // sort_index_vec_multiple_workgroups() function (below).

  // We'll sort the top two bits, and then do sub-sorts on each group.
  dev_mem_t<uword> hist;
  dev_mem_t<eT> A_temp;
  dev_mem_t<uword> out_temp;

  hist.cl_mem_ptr     = get_rt().cl_rt.acquire_memory<uword>(4 * total_num_threads);
  A_temp.cl_mem_ptr   = get_rt().cl_rt.acquire_memory<eT>(n_elem);
  out_temp.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(n_elem);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_elem(n_elem);

  cl_kernel k_count = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_multi_wg_bit_count);
  cl_kernel k_shuffle = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_multi_wg_shuffle);

  // See sort_vec_multiple_workgroups() for a description of the strategy here.
  // This is basically the same, but with slight modifications to handle the
  // fact that our goal is to compute the permutation list for sorting.
  dev_mem_t<eT>* A_in = &A;
  dev_mem_t<uword>* A_index_in = &out;
  dev_mem_t<eT>* A_out = &A_temp;
  dev_mem_t<uword>* A_index_out = &out_temp;

  // Before starting we have to initialize the permutation list.
  linspace<uword>(out, 1, 0, n_elem - 1, n_elem);

  uword step_sort_type = 6 + sort_type; // 6 for ascending, 7 for descending

  // if ascending: bit order [11 10 00 01]
  // if descending: bit order [01 00 10 11]
  uword b = (sizeof(eT) * 8) - 2;
  runtime_t::adapt_uword cl_bit(b);
  runtime_t::adapt_uword cl_sort_type(step_sort_type);
  runtime_t::adapt_uword cl_zero(0);
  runtime_t::adapt_uword cl_in_offset(A_in->cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_index_in_offset(A_index_in->cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_out_offset(A_out->cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_index_out_offset(A_index_out->cl_mem_ptr.offset);

  // Step 1a: count number of each bit pattern for histogramming.
  status  = coot_wrapper(clSetKernelArg)(k_count, 0, sizeof(cl_mem),    &(A_in->cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_count, 1, cl_in_offset.size, cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_count, 2, sizeof(cl_mem),    &(hist.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_count, 3, cl_zero.size,      cl_zero.addr);
  status |= coot_wrapper(clSetKernelArg)(k_count, 4, cl_n_elem.size,    cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k_count, 5, cl_sort_type.size, cl_sort_type.addr);
  status |= coot_wrapper(clSetKernelArg)(k_count, 6, cl_bit.size,       cl_bit.addr);
  coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for bit counting");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_count, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run bit counting kernel");

  // Step 1b: prefix-sum per-workgroup histograms into offsets.
  shifted_prefix_sum(hist, 4 * total_num_threads);

  // Step 2: shuffle points using offsets.
  status  = coot_wrapper(clSetKernelArg)(k_shuffle,  0, sizeof(cl_mem),           &(A_in->cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  1, cl_in_offset.size,        cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  2, sizeof(cl_mem),           &(A_index_in->cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  3, cl_index_in_offset.size,  cl_index_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  4, sizeof(cl_mem),           &(A_out->cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  5, cl_out_offset.size,       cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  6, sizeof(cl_mem),           &(A_index_out->cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  7, cl_index_out_offset.size, cl_index_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  8, sizeof(cl_mem),           &(hist.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k_shuffle,  9, cl_zero.size,             cl_zero.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle, 10, cl_n_elem.size,           cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle, 11, cl_sort_type.size,        cl_sort_type.addr);
  status |= coot_wrapper(clSetKernelArg)(k_shuffle, 12, cl_bit.size,              cl_bit.addr);
  coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for shuffle");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_shuffle, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run shuffle kernel");

  dev_mem_t<eT>* A_tmp = A_in;
  dev_mem_t<uword>* A_index_tmp = A_index_in;
  A_in = A_out;
  A_index_in = A_index_out;
  A_out = A_tmp;
  A_index_out = A_index_tmp;

  // Get the number of negative values vs. the number of positive values.
  const uword first_group_count = (uword) get_val(hist, 2 * total_num_threads);
  const uword second_group_count = n_elem - first_group_count;

  runtime_t::adapt_uword cl_n_elem1(first_group_count);
  runtime_t::adapt_uword cl_n_elem2(second_group_count);
  runtime_t::adapt_uword cl_group2_offset(A_in->cl_mem_ptr.offset + first_group_count);

  // For an ascending sort: the first group (negative numbers) should be sorted descending; the second group (positive numbers) should be sorted ascending.
  // For a descending sort: the first group (positive numbers) should be sorted descending; the second group (negative numbers) should be sorted ascending.
  const uword sort_type1 = 1;
  const uword sort_type2 = 0;
  runtime_t::adapt_uword cl_sort_type1(sort_type1);
  runtime_t::adapt_uword cl_sort_type2(sort_type2);

  // Compute thread and group sizes for the two subgroups.
  size_t total_num_threads1, local_group_size1, total_num_threads2, local_group_size2;
  reduce_kernel_group_info(k_count, first_group_count,  "sort_index_vec", total_num_threads1, local_group_size1);
  reduce_kernel_group_info(k_count, second_group_count, "sort_index_vec", total_num_threads2, local_group_size2);
  const size_t pow2_num_threads1 = next_pow2(total_num_threads1);
  const size_t pow2_num_threads2 = next_pow2(total_num_threads2);
  if (pow2_num_threads1 + pow2_num_threads2 > total_num_threads)
    {
    // Reallocate the histogram vector to the correct size that we'll need.
    get_rt().cl_rt.synchronise();
    get_rt().cl_rt.release_memory(hist.cl_mem_ptr);
    hist.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(4 * (pow2_num_threads1 + pow2_num_threads2));
    }

  const uword hist_offset = 4 * pow2_num_threads1;
  runtime_t::adapt_uword cl_hist_offset(hist_offset);

  // Now perform the rest of the sort on each sub-group (positive or negative).
  for (uword b2 = 0; b2 < sizeof(eT) * 8 - 2; ++b2)
    {
    runtime_t::adapt_uword cl_bit2(b2);

    // Count first group.
    if (first_group_count > 0)
      {
      status  = coot_wrapper(clSetKernelArg)(k_count, 0, sizeof(cl_mem),     &(A_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_count, 1, cl_in_offset.size,  cl_in_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 2, sizeof(cl_mem),     &(hist.cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_count, 3, cl_zero.size,       cl_zero.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 4, cl_n_elem1.size,    cl_n_elem1.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 5, cl_sort_type1.size, cl_sort_type1.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 6, cl_bit2.size,       cl_bit2.addr);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for first bit counting group");

      status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_count, 1, NULL, &pow2_num_threads1, &local_group_size1, 0, NULL, NULL);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run first group bit counting kernel");
      }

    // Count second group.
    if (second_group_count > 0)
      {
      status  = coot_wrapper(clSetKernelArg)(k_count, 0, sizeof(cl_mem),        &(A_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_count, 1, cl_group2_offset.size, cl_group2_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 2, sizeof(cl_mem),        &(hist.cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_count, 3, cl_hist_offset.size,   cl_hist_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 4, cl_n_elem2.size,       cl_n_elem2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 5, cl_sort_type2.size,    cl_sort_type2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_count, 6, cl_bit2.size,          cl_bit2.addr);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for second bit counting group");

      status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_count, 1, NULL, &pow2_num_threads2, &local_group_size2, 0, NULL, NULL);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run second group bit counting kernel");
      }

    // Do shifted prefix-sum on the first group and the second group.
    if (first_group_count > 0)
      {
      shifted_prefix_sum(hist, 4 * pow2_num_threads1);
      }
    if (second_group_count > 0)
      {
      dev_mem_t<uword> hist2{hist.cl_mem_ptr.ptr, hist.cl_mem_ptr.offset + hist_offset};
      shifted_prefix_sum(hist2, 4 * pow2_num_threads2);
      }

    // Now shuffle the first group.
    if (first_group_count > 0)
      {
      status  = coot_wrapper(clSetKernelArg)(k_shuffle,  0, sizeof(cl_mem),           &(A_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  1, cl_in_offset.size,        cl_in_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  2, sizeof(cl_mem),           &(A_index_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  3, cl_index_in_offset.size,  cl_index_in_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  4, sizeof(cl_mem),           &(A_out->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  5, cl_out_offset.size,       cl_out_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  6, sizeof(cl_mem),           &(A_index_out->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  7, cl_index_out_offset.size, cl_index_out_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  8, sizeof(cl_mem),           &(hist.cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  9, cl_zero.size,             cl_zero.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 10, cl_n_elem1.size,          cl_n_elem1.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 11, cl_sort_type1.size,       cl_sort_type1.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 12, cl_bit2.size,             cl_bit2.addr);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for first group shuffle");

      status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_shuffle, 1, NULL, &pow2_num_threads1, &local_group_size1, 0, NULL, NULL);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run shuffle kernel");
      }

    // Shuffle the second group.
    if (second_group_count > 0)
      {
      runtime_t::adapt_uword cl_in_offset2(A_in->cl_mem_ptr.offset + first_group_count);
      runtime_t::adapt_uword cl_index_in_offset2(A_index_in->cl_mem_ptr.offset + first_group_count);
      runtime_t::adapt_uword cl_out_offset2(A_out->cl_mem_ptr.offset + first_group_count);
      runtime_t::adapt_uword cl_index_out_offset2(A_index_out->cl_mem_ptr.offset + first_group_count);

      status  = coot_wrapper(clSetKernelArg)(k_shuffle,  0, sizeof(cl_mem),            &(A_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  1, cl_in_offset2.size,        cl_in_offset2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  2, sizeof(cl_mem),            &(A_index_in->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  3, cl_index_in_offset2.size,  cl_index_in_offset2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  4, sizeof(cl_mem),            &(A_out->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  5, cl_out_offset2.size,       cl_out_offset2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  6, sizeof(cl_mem),            &(A_index_out->cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  7, cl_index_out_offset2.size, cl_index_out_offset2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  8, sizeof(cl_mem),            &(hist.cl_mem_ptr.ptr));
      status |= coot_wrapper(clSetKernelArg)(k_shuffle,  9, cl_hist_offset.size,       cl_hist_offset.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 10, cl_n_elem2.size,           cl_n_elem2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 11, cl_sort_type2.size,        cl_sort_type2.addr);
      status |= coot_wrapper(clSetKernelArg)(k_shuffle, 12, cl_bit2.size,              cl_bit2.addr);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for first group shuffle");

      status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_shuffle, 1, NULL, &pow2_num_threads2, &local_group_size2, 0, NULL, NULL);
      coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run shuffle kernel");
      }

    A_tmp = A_in;
    A_index_tmp = A_index_in;
    A_in = A_out;
    A_index_in = A_index_out;
    A_out = A_tmp;
    A_index_out = A_index_tmp;
    }

  get_rt().synchronise();
  get_rt().cl_rt.release_memory(hist.cl_mem_ptr);
  get_rt().cl_rt.release_memory(A_temp.cl_mem_ptr);
  get_rt().cl_rt.release_memory(out_temp.cl_mem_ptr);
  }



template<typename eT>
inline
void
sort_index_vec_multiple_workgroups(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const size_t total_num_threads, const size_t local_group_size)
  {
  coot_extra_debug_sigprint();

  // First create auxiliary memory.
  dev_mem_t<uword> hist;
  dev_mem_t<eT> A_temp;
  dev_mem_t<uword> out_temp;

  hist.cl_mem_ptr     = get_rt().cl_rt.acquire_memory<uword>(4 * total_num_threads);
  A_temp.cl_mem_ptr   = get_rt().cl_rt.acquire_memory<eT>(n_elem);
  out_temp.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(n_elem);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_elem(n_elem);

  cl_kernel k_count = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_multi_wg_bit_count);
  cl_kernel k_shuffle = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_multi_wg_shuffle);

  // See sort_vec_multiple_workgroups() for a description of the strategy here.
  // This is basically the same, but with slight modifications to handle the
  // fact that our goal is to compute the permutation list for sorting.
  dev_mem_t<eT>* A_in = &A;
  dev_mem_t<uword>* A_index_in = &out;
  dev_mem_t<eT>* A_out = &A_temp;
  dev_mem_t<uword>* A_index_out = &out_temp;

  // Before starting we have to initialize the permutation list.
  linspace<uword>(out, 1, 0, n_elem - 1, n_elem);

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
    runtime_t::adapt_uword cl_zero(0);
    runtime_t::adapt_uword cl_in_offset(A_in->cl_mem_ptr.offset);

    // Step 1a: count number of each bit pattern for histogramming.
    status  = coot_wrapper(clSetKernelArg)(k_count, 0, sizeof(cl_mem),    &(A_in->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_count, 1, cl_in_offset.size, cl_in_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 2, sizeof(cl_mem),    &(hist.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_count, 3, cl_zero.size,      cl_zero.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 4, cl_n_elem.size,    cl_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 5, cl_sort_type.size, cl_sort_type.addr);
    status |= coot_wrapper(clSetKernelArg)(k_count, 6, cl_bit.size,       cl_bit.addr);
    coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for bit counting");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_count, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
    coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run bit counting kernel");

    // Step 1b: prefix-sum per-workgroup histograms into offsets.
    shifted_prefix_sum(hist, 4 * total_num_threads);

    runtime_t::adapt_uword cl_index_in_offset(A_index_in->cl_mem_ptr.offset);
    runtime_t::adapt_uword cl_out_offset(A_out->cl_mem_ptr.offset);
    runtime_t::adapt_uword cl_index_out_offset(A_index_out->cl_mem_ptr.offset);

    // Step 2: shuffle points using offsets.
    status  = coot_wrapper(clSetKernelArg)(k_shuffle,  0, sizeof(cl_mem),           &(A_in->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  1, cl_in_offset.size,        cl_in_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  2, sizeof(cl_mem),           &(A_index_in->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  3, cl_index_in_offset.size,  cl_index_in_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  4, sizeof(cl_mem),           &(A_out->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  5, cl_out_offset.size,       cl_out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  6, sizeof(cl_mem),           &(A_index_out->cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  7, cl_index_out_offset.size, cl_index_out_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  8, sizeof(cl_mem),           &(hist.cl_mem_ptr.ptr));
    status |= coot_wrapper(clSetKernelArg)(k_shuffle,  9, cl_zero.size,             cl_zero.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 10, cl_n_elem.size,           cl_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 11, cl_sort_type.size,        cl_sort_type.addr);
    status |= coot_wrapper(clSetKernelArg)(k_shuffle, 12, cl_bit.size,              cl_bit.addr);
    coot_check_cl_error(status, "coot::opencl::sort_index(): failed to set kernel arguments for shuffle");

    status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_shuffle, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
    coot_check_cl_error(status, "coot::opencl::sort_index(): failed to run shuffle kernel");

    dev_mem_t<eT>* A_tmp = A_in;
    dev_mem_t<uword>* A_index_tmp = A_index_in;
    A_in = A_out;
    A_index_in = A_index_out;
    A_out = A_tmp;
    A_index_out = A_index_tmp;
    }

  get_rt().synchronise();
  get_rt().cl_rt.release_memory(hist.cl_mem_ptr);
  get_rt().cl_rt.release_memory(A_temp.cl_mem_ptr);
  get_rt().cl_rt.release_memory(out_temp.cl_mem_ptr);
  }



template<typename eT>
inline
void
sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const uword stable_sort)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  cl_kernel k;
  if (stable_sort == 0 && sort_type == 0)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_ascending);
    }
  else if (stable_sort == 0 && sort_type == 1)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_descending);
    }
  else if (stable_sort == 1 && sort_type == 0)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_ascending);
    }
  else
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_descending);
    }

  size_t total_num_threads, local_group_size;
  reduce_kernel_group_info(k, n_elem, "sort_index_vec", total_num_threads, local_group_size);
  const size_t pow2_num_threads = next_pow2(total_num_threads);

  // If we will have multiple workgroups, we can't use the single-kernel version.
  if (total_num_threads != local_group_size)
    {
    if (stable_sort == 0 || !is_real<eT>::value)
      {
      sort_index_vec_multiple_workgroups(out, A, n_elem, sort_type, pow2_num_threads, local_group_size);
      }
    else
      {
      stable_sort_index_vec_multiple_workgroups(out, A, n_elem, sort_type, pow2_num_threads, local_group_size);
      }

    return;
    }

  // First, allocate temporary matrices we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);
  dev_mem_t<uword> tmp_mem_index;
  tmp_mem_index.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(n_elem);

  cl_int status = 0;

  const size_t aux_mem_size = (stable_sort == 0) ? (2 * sizeof(uword) * pow2_num_threads) : (4 * sizeof(uword) * pow2_num_threads);

  runtime_t::adapt_uword cl_n_elem(n_elem);
  runtime_t::adapt_uword cl_A_offset(A.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_out_offset(out.cl_mem_ptr.offset);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem),     &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, cl_A_offset.size,   cl_A_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(cl_mem),     &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 3, cl_out_offset.size, cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 4, sizeof(cl_mem),     &(tmp_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 5, sizeof(cl_mem),     &(tmp_mem_index.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 6, cl_n_elem.size,     cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 7, aux_mem_size,       NULL);
  coot_check_cl_error(status, "coot::opencl::sort_index_vec(): failed to set kernel arguments");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, 1, NULL, &pow2_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::sort_index_vec(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  get_rt().cl_rt.release_memory(tmp_mem_index.cl_mem_ptr);
  }
