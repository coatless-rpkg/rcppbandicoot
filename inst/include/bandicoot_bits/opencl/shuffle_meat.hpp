// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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
shuffle(dev_mem_t<eT> out, const uword out_row_offset, const uword out_col_offset, const uword out_M_n_rows,
        const dev_mem_t<eT> in, const uword in_row_offset, const uword in_col_offset, const uword in_M_n_rows,
        const uword n_rows, const uword n_cols, const uword dim)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // dim 0: shuffle the rows of the matrix (or shuffle the elements of a vector)
  // dim 1: shuffle the columns of the matrix
  const uword n_sort_elem = (dim == 0) ? n_rows : n_cols;
  if (n_sort_elem == 0)
    {
    return;
    }
  else if (n_sort_elem == 1)
    {
    // Shortcut: there is nothing to sort, since there is only one element.
    copy_mat(out, in, n_rows, n_cols, out_row_offset, out_col_offset, out_M_n_rows, in_row_offset, in_col_offset, in_M_n_rows);
    return;
    }

  // The variable Philox bijection is only a bijection for powers of 2, so we need to round `n_elem` up.
  const uword n_sort_elem_pow2 = next_pow2(n_sort_elem);
  const uword num_bits = std::log2(n_sort_elem_pow2);

  // Compute size of workgroups.
  size_t kernel_wg_size;
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shuffle);
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): clGetKernelWorkGroupInfo() failed");

  const size_t max_wg_dim_size = get_rt().cl_rt.get_max_wg_dim(0);
  const size_t local_group_size = std::min(std::min(max_wg_dim_size, kernel_wg_size), n_sort_elem_pow2);

  // The variable philox algorithm also needs some random keys.
  dev_mem_t<uword> philox_keys;
  philox_keys.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(24);
  fill_randu<uword>(philox_keys, 24);

  // Note that all vectors are treated as column vectors and passed with dim == 0 by convention.
  const uword in_offset  =  in_row_offset +  in_col_offset *  in_M_n_rows;
  const uword out_offset = out_row_offset + out_col_offset * out_M_n_rows;

  const uword elems_per_elem =  (dim == 0) ? n_cols       : n_rows;
  const uword in_elem_stride  = (dim == 0) ? in_M_n_rows  : 1;
  const uword out_elem_stride = (dim == 0) ? out_M_n_rows : 1;
  const uword in_incr         = (dim == 0) ? 1            : in_M_n_rows;
  const uword out_incr        = (dim == 0) ? 1            : out_M_n_rows;

  // Split to the correct implementation: if the number of threads is too much, we have to use multiple kernels.
  if (size_t(n_sort_elem_pow2) == local_group_size)
    {
    shuffle_small(out, out_offset, out_incr, out_elem_stride, in, in_offset, in_incr, in_elem_stride, n_sort_elem, elems_per_elem, n_sort_elem_pow2, num_bits, philox_keys);
    }
  else
    {
    shuffle_large(out, out_offset, out_incr, out_elem_stride, in, in_offset, in_incr, in_elem_stride, n_sort_elem, elems_per_elem, n_sort_elem_pow2, num_bits, local_group_size, philox_keys);
    }

  get_rt().cl_rt.release_memory(philox_keys.cl_mem_ptr);
  }



template<typename eT>
inline
void
shuffle_small(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const dev_mem_t<uword> philox_keys)
  {
  coot_extra_debug_sigprint();

  runtime_t::adapt_uword cl_out_offset(out.cl_mem_ptr.offset + out_offset);
  runtime_t::adapt_uword cl_out_incr(out_incr);
  runtime_t::adapt_uword cl_out_elem_stride(out_elem_stride);
  runtime_t::adapt_uword cl_in_offset(in.cl_mem_ptr.offset + in_offset);
  runtime_t::adapt_uword cl_in_incr(in_incr);
  runtime_t::adapt_uword cl_in_elem_stride(in_elem_stride);
  runtime_t::adapt_uword cl_n_elem(n_elem);
  runtime_t::adapt_uword cl_elems_per_elem(elems_per_elem);
  runtime_t::adapt_uword cl_n_elem_pow2(n_elem_pow2);
  runtime_t::adapt_uword cl_num_bits(num_bits);

  cl_int status = 0;
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shuffle);

  status  = coot_wrapper(clSetKernelArg)(k,  0, sizeof(cl_mem),               &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k,  1, cl_out_offset.size,           cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  2, cl_out_incr.size,             cl_out_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  3, cl_out_elem_stride.size,      cl_out_elem_stride.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  4, sizeof(cl_mem),               &(in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k,  5, cl_in_offset.size,            cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  6, cl_in_incr.size,              cl_in_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  7, cl_in_elem_stride.size,       cl_in_elem_stride.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  8, cl_n_elem.size,               cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k,  9, cl_elems_per_elem.size,       cl_elems_per_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 10, cl_n_elem_pow2.size,          cl_n_elem_pow2.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 11, sizeof(cl_mem),               &(philox_keys.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 12, cl_num_bits.size,             cl_num_bits.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 13, cl_n_elem.size * n_elem_pow2, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): clSetKernelArg() failed");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, 1, NULL, &n_elem_pow2, &n_elem_pow2, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): couldn't execute kernel");
  }



template<typename eT>
inline
void
shuffle_large(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const size_t local_group_size, const dev_mem_t<uword> philox_keys)
  {
  coot_extra_debug_sigprint();

  runtime_t::adapt_uword cl_out_offset(out.cl_mem_ptr.offset + out_offset);
  runtime_t::adapt_uword cl_out_incr(out_incr);
  runtime_t::adapt_uword cl_out_elem_stride(out_elem_stride);
  runtime_t::adapt_uword cl_in_offset(in.cl_mem_ptr.offset + in_offset);
  runtime_t::adapt_uword cl_in_incr(in_incr);
  runtime_t::adapt_uword cl_in_elem_stride(in_elem_stride);
  runtime_t::adapt_uword cl_n_elem(n_elem);
  runtime_t::adapt_uword cl_elems_per_elem(elems_per_elem);
  runtime_t::adapt_uword cl_n_elem_pow2(n_elem_pow2);
  runtime_t::adapt_uword cl_num_bits(num_bits);

  const size_t total_num_threads = size_t(n_elem_pow2);

  // First, compute the locations where everything will map to.
  const uword aux_size = total_num_threads / local_group_size; // remember total_num_threads and local_group_size are both powers of 2
  dev_mem_t<uword> out_block_mem;
  out_block_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(aux_size);

  cl_int status = 0;
  cl_kernel k1 = get_rt().cl_rt.get_kernel(zeroway_kernel_id::shuffle_large_compute_locs);

  status  = coot_wrapper(clSetKernelArg)(k1, 0, sizeof(cl_mem),                    &(out_block_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k1, 1, cl_n_elem.size,                    cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k1, 2, cl_n_elem_pow2.size,               cl_n_elem_pow2.addr);
  status |= coot_wrapper(clSetKernelArg)(k1, 3, sizeof(cl_mem),                    &(philox_keys.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k1, 4, cl_num_bits.size,                  cl_num_bits.addr);
  status |= coot_wrapper(clSetKernelArg)(k1, 5, cl_n_elem.size * local_group_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): clSetKernelArg() failed");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k1, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): computing shuffle locations kernel failed");

  // Now prefix-sum the locations so we have usable offsets.
  shifted_prefix_sum(out_block_mem, aux_size);

  // Finally, run the second kernel that actually does the shuffle.
  cl_kernel k2 = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shuffle_large);

  status  = coot_wrapper(clSetKernelArg)(k2,  0, sizeof(cl_mem),                    &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k2,  1, cl_out_offset.size,                cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  2, cl_out_incr.size,                  cl_out_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  3, cl_out_elem_stride.size,           cl_out_elem_stride.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  4, sizeof(cl_mem),                    &(in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k2,  5, cl_in_offset.size,                 cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  6, cl_in_incr.size,                   cl_in_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  7, cl_in_elem_stride.size,            cl_in_elem_stride.addr);
  status |= coot_wrapper(clSetKernelArg)(k2,  8, sizeof(cl_mem),                    &(out_block_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k2,  9, cl_n_elem.size,                    cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k2, 10, cl_elems_per_elem.size,            cl_elems_per_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k2, 11, cl_n_elem_pow2.size,               cl_n_elem_pow2.addr);
  status |= coot_wrapper(clSetKernelArg)(k2, 12, sizeof(cl_mem),                    &(philox_keys.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(k2, 13, cl_num_bits.size,                  cl_num_bits.addr);
  status |= coot_wrapper(clSetKernelArg)(k2, 14, cl_n_elem.size * local_group_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): clSetKernelArg() failed");

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k2, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shuffle(): shuffle kernel failed");

  get_rt().cl_rt.release_memory(out_block_mem.cl_mem_ptr);
  }
