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

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::approx_equal(): CUDA runtime not valid" );

  // We will do a two-array reduce into a u32 array; from there, we can use any().
  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::approx_equal);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  const uword n_elem = n_rows * n_cols;
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<u32> aux(dims.d[0], 1);
  dev_mem_t<u32> aux_mem = aux.get_dev_mem(false);

  // Ensure we always use a power of 2 for the number of threads.
  const uword num_threads = next_pow2(dims.d[3]);

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');
  uword mode = 0;
  if (do_abs)
    mode += 1;
  if (do_rel)
    mode += 2;

  const uword A_offset = A_row_offset + A_col_offset * A_M_n_rows;
  const uword B_offset = B_row_offset + B_col_offset * B_M_n_rows;
  const eT* A_mem_ptr = A.cuda_mem_ptr + A_offset;
  const eT* B_mem_ptr = B.cuda_mem_ptr + B_offset;

  const void* args[] = {
      &(aux_mem.cuda_mem_ptr),
      &(A_mem_ptr),
      (uword*) &A_M_n_rows,
      &(B_mem_ptr),
      (uword*) &B_M_n_rows,
      (uword*) &n_rows,
      (uword*) &n_elem,
      (uword*) &mode,
      (eT*) &abs_tol,
      (eT*) &rel_tol };

  CUresult curesult = coot_wrapper(cuLaunchKernel)(
        num_threads <= 32 ? k_small : k, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
        dims.d[0], dims.d[1], dims.d[2],
        num_threads, dims.d[4], dims.d[5],
        num_threads * sizeof(u32), // shared mem should have size equal to number of threads
        NULL,
        (void**) args,
        0);

  coot_check_cuda_error(curesult, "coot::cuda::approx_equal(): cuLaunchKernel() failed");

  if (aux.n_elem == 1)
    {
    return (aux[0] == 0) ? false : true;
    }
  else
    {
    // Perform an and-reduce.
    CUfunction second_k = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce);
    CUfunction second_k_small = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce_small);

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

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::approx_equal_cube(): CUDA runtime not valid" );

  // We will do a two-array reduce into a u32 array; from there, we can use any().
  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_cube);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::approx_equal_cube_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  const uword n_elem = n_rows * n_cols * n_slices;
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<u32> aux(dims.d[0], 1);
  dev_mem_t<u32> aux_mem = aux.get_dev_mem(false);

  // Ensure we always use a power of 2 for the number of threads.
  const uword num_threads = next_pow2(dims.d[3]);

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');
  uword mode = 0;
  if (do_abs)
    mode += 1;
  if (do_rel)
    mode += 2;

  const uword A_offset = A_row_offset + A_col_offset * A_M_n_rows + A_slice_offset * A_M_n_rows * A_M_n_cols;
  const uword B_offset = B_row_offset + B_col_offset * B_M_n_rows + B_slice_offset * B_M_n_rows * B_M_n_cols;
  const eT* A_mem_ptr = A.cuda_mem_ptr + A_offset;
  const eT* B_mem_ptr = B.cuda_mem_ptr + B_offset;

  const void* args[] = {
      &(aux_mem.cuda_mem_ptr),
      &(A_mem_ptr),
      (uword*) &A_M_n_rows,
      (uword*) &A_M_n_cols,
      &(B_mem_ptr),
      (uword*) &B_M_n_rows,
      (uword*) &B_M_n_cols,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_elem,
      (uword*) &mode,
      (eT*) &abs_tol,
      (eT*) &rel_tol };

  CUresult curesult = coot_wrapper(cuLaunchKernel)(
        num_threads <= 32 ? k_small : k, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
        dims.d[0], dims.d[1], dims.d[2],
        num_threads, dims.d[4], dims.d[5],
        num_threads * sizeof(u32), // shared mem should have size equal to number of threads
        NULL,
        (void**) args,
        0);

  coot_check_cuda_error(curesult, "coot::cuda::approx_equal(): cuLaunchKernel() failed");

  if (aux.n_elem == 1)
    {
    return (aux[0] == 0) ? false : true;
    }
  else
    {
    // Perform an and-reduce.
    CUfunction second_k = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce);
    CUfunction second_k_small = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce_small);

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
