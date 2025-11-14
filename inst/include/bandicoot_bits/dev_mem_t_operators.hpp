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



inline
bool
operator==(const coot_cl_mem& a, const coot_cl_mem& b)
  {
  return a.ptr == b.ptr && a.offset == b.offset;
  }



inline
bool
operator!=(const coot_cl_mem& a, const coot_cl_mem& b)
  {
  return !(a == b);
  }



inline
coot_cl_mem
operator+(const coot_cl_mem& a, size_t b)
  {
  return coot_cl_mem { a.ptr, a.offset + b };
  }



inline
coot_cl_mem
operator-(const coot_cl_mem& a, size_t b)
  {
  // limit offset to 0
  return coot_cl_mem { a.ptr, (b > a.offset) ? 0 : a.offset - b };
  }



inline
coot_cl_mem&
operator+=(coot_cl_mem& a, size_t b)
  {
  a.offset += b;
  return a;
  }



inline
coot_cl_mem&
operator-=(coot_cl_mem& a, size_t b)
  {
  a.offset = (b > a.offset) ? 0 : a.offset - b;
  return a;
  }



template<typename eT>
inline
bool
operator==(const dev_mem_t<eT>& a, const dev_mem_t<eT>& b)
  {
  // Regardless of backend we can just do a full equality check here (since the offset will be 0 for CUDA devices).
  return (a.cl_mem_ptr == b.cl_mem_ptr);
  }



template<typename eT>
inline
bool
operator!=(const dev_mem_t<eT>& a, const dev_mem_t<eT>& b)
  {
  return (a.cl_mem_ptr != b.cl_mem_ptr);
  }



template<typename eT>
inline
dev_mem_t<eT>
operator+(const dev_mem_t<eT>& a, const size_t b)
  {
  dev_mem_t<eT> result({ NULL, 0 });
  if (get_rt().backend == CL_BACKEND)
    {
    result.cl_mem_ptr = a.cl_mem_ptr + b;
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    result.cuda_mem_ptr = a.cuda_mem_ptr + b;
    }

  return result;
  }



template<typename eT>
inline
dev_mem_t<eT>
operator-(const dev_mem_t<eT>& a, const size_t b)
  {
  dev_mem_t<eT> result({ NULL, 0 });
  if (get_rt().backend == CL_BACKEND)
    {
    result.cl_mem_ptr = a.cl_mem_ptr - b;
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    result.cuda_mem_ptr = a.cuda_mem_ptr - b;
    }

  return result;
  }



template<typename eT>
inline
dev_mem_t<eT>&
operator+=(dev_mem_t<eT>& a, const size_t b)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    a.cl_mem_ptr += b;
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    a.cuda_mem_ptr += b;
    }

  return a;
  }



template<typename eT>
inline
dev_mem_t<eT>&
operator-=(dev_mem_t<eT>& a, const size_t b)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    a.cl_mem_ptr -= b;
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    a.cuda_mem_ptr -= b;
    }

  return a;
  }
