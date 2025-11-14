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



// NOTE: this is not exactly prefix-sum as typically taught.  It returns a result that is shifted by one element.
// An input of [1, 3, 2, 4] returns an output of [0, 1, 4, 6]---*not* the "typical" output of [1, 4, 6, 10].



template<typename eT>
inline
void
shifted_prefix_sum_small(dev_mem_t<eT> mem, const uword offset, const uword n_elem, const kernel_dims& dims)
  {
  coot_extra_debug_sigprint();

  typedef typename cuda_type<eT>::type ceT;
  ceT* mem_ptr = mem.cuda_mem_ptr + offset;

  const void* args[] = { &mem_ptr, (uword*) &n_elem };

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_small);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      2 * dims.d[3] * sizeof(eT),
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::shifted_prefix_sum_small(): cuLaunchKernel() failed");
  }



template<typename eT>
inline
void
shifted_prefix_sum_large(dev_mem_t<eT> mem, const uword offset, const uword n_elem, const kernel_dims& dims)
  {
  coot_extra_debug_sigprint();

  // For arrays larger than we can handle in a single block, we operate recursively:
  // we perform up-sweeps until it fits in a single block;
  // then we prefix-sum the single block with shifted_prefix_sum_small();
  // then we perform down-sweeps until the entire array is prefix-summed.

  typedef typename cuda_type<eT>::type ceT;
  ceT* mem_ptr = mem.cuda_mem_ptr + offset;

  Col<eT> tmp(dims.d[0]);
  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_subgroups);
  const void* args1[] = { &mem_ptr, &(tmp_mem.cuda_mem_ptr), (uword*) &n_elem };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k1,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      2 * dims.d[3] * sizeof(eT),
      NULL,
      (void**) args1,
      0);

  coot_check_cuda_error(result, "coot::cuda::shifted_prefix_sum_large(): cuLaunchKernel() failed for sum_subgroups");

  // Now prefix-sum the result memory recursively before we take a second pass to add offsets to every workgroup's memory.
  // After this, tmp_mem will properly hold shifted-prefix-summed memory.
  shifted_prefix_sum(tmp_mem, 0, dims.d[0]);

  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_add_offset);
  const void* args2[] = { &mem_ptr, &(tmp_mem.cuda_mem_ptr), (uword*) &n_elem };

  result = coot_wrapper(cuLaunchKernel)(
      k2,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      2 * dims.d[3] * sizeof(eT),
      NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result, "coot::cuda::shifted_prefix_sum_large(): cuLaunchKernel() failed for add_offset");
  }


template<typename eT>
inline
void
shifted_prefix_sum(dev_mem_t<eT> mem, const uword offset, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return;
    }
  else if (n_elem == 1)
    {
    // Set the memory to 0.
    set_val(mem, 0, eT(0));
    return;
    }

  // Compute the number of threads we need to handle an array of this size.
  // Each thread will handle 2 memory elements.
  const size_t pow2_num_threads = next_pow2(n_elem / 2);
  const kernel_dims dims = one_dimensional_grid_dims(pow2_num_threads);

  if (dims.d[0] == 1)
    {
    shifted_prefix_sum_small(mem, offset, n_elem, dims);
    }
  else
    {
    shifted_prefix_sum_large(mem, offset, n_elem, dims);
    }
  }
