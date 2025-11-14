// Copyright 2023-2025 Ryan Curtin (http://www.ratml.org)
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
 * Return `true` if the two matrices' elements are all approximately equal.
 */
template<typename eT>
inline
bool
approx_equal(const dev_mem_t<eT> A,
             const uword A_row_offset,
             const uword A_col_offset,
             const uword A_M_n_rows,
             const dev_mem_t<eT> B,
             const uword B_row_offset,
             const uword B_col_offset,
             const uword B_M_n_rows,
             const uword n_rows,
             const uword n_cols,
             const char sig,
             const eT abs_tol,
             const eT rel_tol)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::approx_equal_cube(): OpenCL runtime not valid" );

  cl_int status = 0;

  // We will do a two-array reduce into a u32 array; from there, we can use any().
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::approx_equal);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_small);

  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::approx_equal(): could not get kernel workgroup size");

  const uword n_elem = n_rows * n_cols;
  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword subgroup_size = get_rt().cl_rt.get_subgroup_size();

  uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // Create auxiliary memory.
  const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  Mat<u32> aux(aux_size, 1);
  dev_mem_t<u32> aux_mem = aux.get_dev_mem(false);

  runtime_t::cq_guard guard;

  const uword A_offset = A_row_offset + A_col_offset * A_M_n_rows + A.cl_mem_ptr.offset;
  const uword B_offset = B_row_offset + B_col_offset * B_M_n_rows + B.cl_mem_ptr.offset;

  runtime_t::adapt_uword dev_A_offset(A_offset);
  runtime_t::adapt_uword dev_B_offset(B_offset);
  runtime_t::adapt_uword dev_A_M_n_rows(A_M_n_rows);
  runtime_t::adapt_uword dev_B_M_n_rows(B_M_n_rows);
  runtime_t::adapt_uword dev_n_rows(n_rows);
  runtime_t::adapt_uword dev_n_elem(n_elem);

  // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
  const uword pow2_group_size = next_pow2(local_group_size);
  const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');

  uword mode = 0;
  if (do_abs == true)
    mode += 1;
  if (do_rel == true)
    mode += 2;
  runtime_t::adapt_uword dev_mode(mode);

  // If the number of threads is less than the subgroup size, we need to use the small kernel.
  cl_kernel* k_use = (pow2_group_size <= subgroup_size) ? &k_small : &k;

  typedef typename cl_type<eT>::type ceT;
  ceT cl_abs_tol = to_cl_type(abs_tol);
  ceT cl_rel_tol = to_cl_type(rel_tol);

  status |= coot_wrapper(clSetKernelArg)(*k_use,  0, sizeof(cl_mem),                &(aux_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  1, sizeof(cl_mem),                &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  2, dev_A_offset.size,             dev_A_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  3, dev_A_M_n_rows.size,           dev_A_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  4, sizeof(cl_mem),                &(B.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  5, dev_B_offset.size,             dev_B_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  6, dev_B_M_n_rows.size,           dev_B_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  7, dev_n_rows.size,               dev_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  8, dev_n_elem.size,               dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  9, sizeof(u32) * pow2_group_size, NULL);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 10, dev_mode.size,                 dev_mode.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 11, sizeof(ceT),                   &cl_abs_tol);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 12, sizeof(ceT),                   &cl_rel_tol);
  coot_check_cl_error(status, "coot::opencl::approx_equal(): could not set kernel arguments");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::approx_equal(): could not run kernel");

  if (aux.n_elem == 1)
    {
    return (aux[0] == 0) ? false : true;
    }
  else
    {
    // Perform an and-reduce.
    cl_kernel second_k = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce);
    cl_kernel second_k_small = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce_small);

    u32 result = generic_reduce<u32, u32>(aux_mem,
                                          aux.n_elem,
                                          "approx_equal",
                                          second_k,
                                          second_k_small,
                                          std::make_tuple(/* no extra args */),
                                          second_k,
                                          second_k_small,
                                          std::make_tuple(/* no extra args */));

    return (result == 0) ? false : true;
    }
  }



/**
 * Return `true` if the two cubes' elements are all approximately equal.
 */
template<typename eT>
inline
bool
approx_equal_cube(const dev_mem_t<eT> A,
                  const uword A_row_offset,
                  const uword A_col_offset,
                  const uword A_slice_offset,
                  const uword A_M_n_rows,
                  const uword A_M_n_cols,
                  const dev_mem_t<eT> B,
                  const uword B_row_offset,
                  const uword B_col_offset,
                  const uword B_slice_offset,
                  const uword B_M_n_rows,
                  const uword B_M_n_cols,
                  const uword n_rows,
                  const uword n_cols,
                  const uword n_slices,
                  const char sig,
                  const eT abs_tol,
                  const eT rel_tol)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::approx_equal_cube(): OpenCL runtime not valid" );

  cl_int status = 0;

  // We will do a two-array reduce into a u32 array; from there, we can use any().
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_cube);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_cube_small);

  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::approx_equal_cube(): could not get kernel workgroup size");

  const uword n_elem = n_rows * n_cols * n_slices;
  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword subgroup_size = get_rt().cl_rt.get_subgroup_size();

  uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // Create auxiliary memory.
  const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  Mat<u32> aux(aux_size, 1);
  dev_mem_t<u32> aux_mem = aux.get_dev_mem(false);

  runtime_t::cq_guard guard;

  const uword A_offset = A_row_offset + A_col_offset * A_M_n_rows + A_slice_offset * A_M_n_rows * A_M_n_cols + A.cl_mem_ptr.offset;
  const uword B_offset = B_row_offset + B_col_offset * B_M_n_rows + B_slice_offset * B_M_n_rows * B_M_n_cols + B.cl_mem_ptr.offset;

  runtime_t::adapt_uword dev_A_offset(A_offset);
  runtime_t::adapt_uword dev_B_offset(B_offset);
  runtime_t::adapt_uword dev_A_M_n_rows(A_M_n_rows);
  runtime_t::adapt_uword dev_B_M_n_rows(B_M_n_rows);
  runtime_t::adapt_uword dev_A_M_n_cols(A_M_n_cols);
  runtime_t::adapt_uword dev_B_M_n_cols(B_M_n_cols);
  runtime_t::adapt_uword dev_n_rows(n_rows);
  runtime_t::adapt_uword dev_n_cols(n_cols);
  runtime_t::adapt_uword dev_n_elem(n_elem);

  // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
  const uword pow2_group_size = next_pow2(local_group_size);
  const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');

  uword mode = 0;
  if (do_abs == true)
    mode += 1;
  if (do_rel == true)
    mode += 2;
  runtime_t::adapt_uword dev_mode(mode);

  typedef typename cl_type<eT>::type ceT;
  ceT cl_abs_tol = to_cl_type(abs_tol);
  ceT cl_rel_tol = to_cl_type(rel_tol);

  // If the number of threads is less than the subgroup size, we need to use the small kernel.
  cl_kernel* k_use = (pow2_group_size <= subgroup_size) ? &k_small : &k;

  status |= coot_wrapper(clSetKernelArg)(*k_use,  0, sizeof(cl_mem),                &(aux_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  1, sizeof(cl_mem),                &(A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  2, dev_A_offset.size,             dev_A_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  3, dev_A_M_n_rows.size,           dev_A_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  4, dev_A_M_n_cols.size,           dev_A_M_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  5, sizeof(cl_mem),                &(B.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use,  6, dev_B_offset.size,             dev_B_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  7, dev_B_M_n_rows.size,           dev_B_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  8, dev_B_M_n_cols.size,           dev_B_M_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use,  9, dev_n_rows.size,               dev_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 10, dev_n_cols.size,               dev_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 11, dev_n_elem.size,               dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 12, sizeof(u32) * pow2_group_size, NULL);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 13, dev_mode.size,                 dev_mode.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 14, sizeof(ceT),                   &cl_abs_tol);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 15, sizeof(ceT),                   &cl_rel_tol);
  coot_check_cl_error(status, "coot::opencl::approx_equal_cube(): could not set kernel arguments");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::approx_equal_cube(): could not run kernel");

  if (aux.n_elem == 1)
    {
    return (aux[0] == 0) ? false : true;
    }
  else
    {
    // Perform an and-reduce.
    cl_kernel second_k = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce);
    cl_kernel second_k_small = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce_small);

    u32 result = generic_reduce<u32, u32>(aux_mem,
                                          aux.n_elem,
                                          "approx_equal_cube",
                                          second_k,
                                          second_k_small,
                                          std::make_tuple(/* no extra args */),
                                          second_k,
                                          second_k_small,
                                          std::make_tuple(/* no extra args */));

    return (result == 0) ? false : true;
    }
  }
