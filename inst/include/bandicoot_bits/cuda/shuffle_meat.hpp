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

  // dim 0: shuffle the rows of the matrix
  // dim 1: shuffle the columns of the matrix (or shuffle the elements of a vector)
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
  const kernel_dims dims = one_dimensional_grid_dims(n_sort_elem_pow2);
  const uword num_bits = std::log2(n_sort_elem_pow2);

  // The variable philox algorithm also needs some random keys.
  dev_mem_t<uword> philox_keys;
  philox_keys.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(24);
  curandStatus_t result = coot_wrapper(curandGenerate)(get_rt().cuda_rt.xorwow_rand, (u32*) philox_keys.cuda_mem_ptr, 48);
  coot_check_curand_error(result, "coot::cuda::shuffle(): curandGenerate() failed");

  // Note that all vectors are treated as column vectors and passed with dim == 0 by convention.
  const uword in_offset  =  in_row_offset +  in_col_offset *  in_M_n_rows;
  const uword out_offset = out_row_offset + out_col_offset * out_M_n_rows;

  const uword elems_per_elem =  (dim == 0) ? n_cols       : n_rows;
  const uword in_elem_stride  = (dim == 0) ? in_M_n_rows  : 1;
  const uword out_elem_stride = (dim == 0) ? out_M_n_rows : 1;
  const uword in_incr         = (dim == 0) ? 1            : in_M_n_rows;
  const uword out_incr        = (dim == 0) ? 1            : out_M_n_rows;

  if (dims.d[0] == 1)
    {
    shuffle_small(out, out_offset, out_incr, out_elem_stride, in, in_offset, in_incr, in_elem_stride, n_sort_elem, elems_per_elem, n_sort_elem_pow2, num_bits, dims, philox_keys);
    }
  else
    {
    shuffle_large(out, out_offset, out_incr, out_elem_stride, in, in_offset, in_incr, in_elem_stride, n_sort_elem, elems_per_elem, n_sort_elem_pow2, num_bits, dims, philox_keys);
    }

  get_rt().cuda_rt.release_memory(philox_keys.cuda_mem_ptr);
  }



template<typename eT>
inline
void
shuffle_small(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const kernel_dims& dims, const dev_mem_t<uword> philox_keys)
  {
  coot_extra_debug_sigprint();

  typedef typename cuda_type<eT>::type ceT;

        ceT* out_ptr = out.cuda_mem_ptr + out_offset;
  const ceT* in_ptr  =  in.cuda_mem_ptr + in_offset;

  const void* args[] = {
    &out_ptr,
    (uword*) &out_incr,
    (uword*) &out_elem_stride,
    &in_ptr,
    (uword*) &in_incr,
    (uword*) &in_elem_stride,
    (uword*) &n_elem,
    (uword*) &elems_per_elem,
    (uword*) &n_elem_pow2,
    &(philox_keys.cuda_mem_ptr),
    (uword*) &num_bits };

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::shuffle);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      sizeof(uword) * n_elem_pow2, // shared mem should have size one uword per thread
      NULL,
      (void**) args,
      0);
  coot_check_cuda_error(result, "coot::cuda::shuffle(): cuLaunchKernel() failed");
  }



template<typename eT>
inline
void
shuffle_large(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const kernel_dims& dims, const dev_mem_t<uword> philox_keys)
  {
  coot_extra_debug_sigprint();

  typedef typename cuda_type<eT>::type ceT;

        ceT* out_ptr = out.cuda_mem_ptr + out_offset;
  const ceT* in_ptr  =  in.cuda_mem_ptr + in_offset;

  dev_mem_t<uword> out_block_mem;
  out_block_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(dims.d[0]);

  const void* args1[] = {
    &(out_block_mem.cuda_mem_ptr),
    (uword*) &n_elem,
    (uword*) &n_elem_pow2,
    &(philox_keys.cuda_mem_ptr),
    (uword*) &num_bits };

  CUfunction k1 = get_rt().cuda_rt.get_kernel(zeroway_kernel_id::shuffle_large_compute_locs);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k1,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      sizeof(uword) * dims.d[3], // shared mem should have size one uword per thread
      NULL,
      (void**) args1,
      0);

  coot_check_cuda_error(result, "coot::cuda::shuffle(): cuLaunchKernel() failed for location computation");

  // Now we have to prefix-sum the block memory.
  shifted_prefix_sum(out_block_mem, 0, dims.d[0]);

  // Finally, we can shuffle everything correctly.
  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::shuffle_large);

  const void* args2[] = {
    &out_ptr,
    (uword*) &out_incr,
    (uword*) &out_elem_stride,
    &in_ptr,
    (uword*) &in_incr,
    (uword*) &in_elem_stride,
    &(out_block_mem.cuda_mem_ptr),
    (uword*) &n_elem,
    (uword*) &elems_per_elem,
    (uword*) &n_elem_pow2,
    &(philox_keys.cuda_mem_ptr),
    (uword*) &num_bits };

  result = coot_wrapper(cuLaunchKernel)(
      k2,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      sizeof(uword) * dims.d[3],
      NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result, "coot::cuda::shuffle() cuLaunchKernel() failed for shuffle");

  get_rt().cuda_rt.release_memory(out_block_mem.cuda_mem_ptr);
  }
