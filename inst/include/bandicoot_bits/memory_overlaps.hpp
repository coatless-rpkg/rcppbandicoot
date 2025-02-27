// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (https://www.ratml.org)
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



template<typename eT1, typename eT2>
inline
bool
mem_overlaps(const dev_mem_t<eT1>& a,
             const uword a_offset,
             const uword a_n_elem,
             const dev_mem_t<eT2> b,
             const uword b_offset,
             const uword b_n_elem)
  {
  if (a.cl_mem_ptr.ptr == nullptr || b.cl_mem_ptr.ptr == nullptr)
    {
    return false; // if memory is not initialized, it's not overlapping
    }

  if (a_n_elem == 0 || b_n_elem == 0)
    {
    return false; // empty matrices cannot overlap
    }

  // OpenCL pointer arithmetic is not allowed---so any time something is an alias,
  // it will have the same pointer.
  //
  // If we are using the CUDA backend, then pointer arithmetic is allowed.
  if (get_rt().backend == CL_BACKEND)
    {
    if (a.cl_mem_ptr.ptr != b.cl_mem_ptr.ptr)
      {
      return false;
      }

    // If it is the same pointer, check if the range overlaps.
    const size_t a_start = a_offset;
    const size_t a_end = a_offset + a_n_elem - 1;
    const size_t b_start = b_offset;
    const size_t b_end = b_offset + b_n_elem - 1;

    return (a_start <= b_end) && (a_end >= b_start);
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    const eT1* a_start = ((eT1*) a.cuda_mem_ptr) + a_offset;
    const eT1* a_end = ((eT1*) a.cuda_mem_ptr) + a_offset + a_n_elem - 1;
    const eT2* b_start = ((eT2*) b.cuda_mem_ptr) + b_offset;
    const eT2* b_end = ((eT2*) b.cuda_mem_ptr) + b_offset + b_n_elem - 1;

    return ((void*) a_start <= (void*) b_end) && ((void*) a_end >= (void*) b_start);
    }

  return false; // fix compilation warning
  }
