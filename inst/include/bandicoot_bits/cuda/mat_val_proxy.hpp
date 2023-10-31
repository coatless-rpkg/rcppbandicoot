// Copyright 2019 Ryan Curtin (http://ratml.org)
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



// Utility functions for MatValProxy with the CUDA backend.

template<typename eT>
inline
eT
get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  // We'll just use cudaMemcpy() to copy back the single value.
  // This is inefficient, but without using Unified Memory, I don't see
  // an alternative.

  eT val = eT(0);

  cudaError_t status = coot_wrapper(cudaMemcpy)((void*) &val,
                                                (void*) (mem.cuda_mem_ptr + index),
                                                sizeof(eT),
                                                cudaMemcpyDeviceToHost);

  coot_check_cuda_error(status, "coot::cuda::get_val(): couldn't access device memory");

  return val;
  }



template<typename eT>
inline
void
set_val(dev_mem_t<eT> mem, const uword index, const eT in_val)
  {
  coot_extra_debug_sigprint();

  // We'll just use cudaMemcpy() to copy over the single value.

  cudaError_t status = coot_wrapper(cudaMemcpy)((void*) (mem.cuda_mem_ptr + index),
                                                (void*) &in_val,
                                                sizeof(eT),
                                                cudaMemcpyHostToDevice);

  coot_check_cuda_error(status, "coot::cuda::set_val(): couldn't access device memory");
  }



template<typename eT>
inline
void
val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  // We'll run a kernel with only one worker to update the index.
  eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
             mem, mem,
             val, (eT) 0,
             1, 1,
             index, 0, index + 1,
             index, 0, index + 1);
  }



template<typename eT>
inline
void
val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  // We'll run a kernel with only one worker to update the index.
  eop_scalar(twoway_kernel_id::equ_array_minus_scalar_post,
             mem, mem,
             val, (eT) 0,
             1, 1,
             index, 0, index + 1,
             index, 0, index + 1);
  }



template<typename eT>
inline
void
val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  // We'll run a kernel with only one worker to update the index.
  eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
             mem, mem,
             val, (eT) 1,
             1, 1,
             index, 0, index + 1,
             index, 0, index + 1);
  }



template<typename eT>
inline
void
val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  // We'll run a kernel with only one worker to update the index.
  eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
             mem, mem,
             val, (eT) 1,
             1, 1,
             index, 0, index + 1,
             index, 0, index + 1);
  }
