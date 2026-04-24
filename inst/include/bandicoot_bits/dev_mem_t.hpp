// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
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



// OpenCL kernels cannot use a cl_mem that is offset from an original cl_mem;
// all aliases must be passed to a kernel as both a cl_mem and a size_t offset.
// So unfortunately we have to hold two values for OpenCL memory.
struct coot_cl_mem
  {
  cl_mem ptr;
  size_t offset;
  };



// Vulkan memory is represented as a buffer coupled with its backing device
// memory and an offset.
struct coot_vk_mem
  {
  VkBuffer buffer;
  VkDeviceMemory memory;
  size_t offset;
  size_t byte_offset;
  size_t byte_size;
  };



// This can hold either CUDA memory, CL memory, or Vulkan memory.
template<typename eT>
union dev_mem_t
  {
  coot_cl_mem cl_mem_ptr;
  typename cuda_type<eT>::type* cuda_mem_ptr;
  coot_vk_mem vk_mem_ptr;

  // Manual overloading when we set a CUDA pointer: ensure the last bytes of dev_mem_t are 0 so that comparisons work.
  dev_mem_t& operator=(const eT* cuda_other_mem)
    {
    cl_mem_ptr.offset = 0;
    cuda_mem_ptr = cuda_other_mem;
    }
  };
