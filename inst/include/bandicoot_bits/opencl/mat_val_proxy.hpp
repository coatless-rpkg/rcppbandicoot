// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
// ------------------------------------------------------------------------



// Utility functions for MatValProxy with the OpenCL backend.

template<typename eT>
inline
eT
get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(
      get_rt().cl_rt.get_cq(),
      mem.cl_mem_ptr.ptr,
      CL_TRUE,
      CL_MAP_READ,
      sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
      sizeof(ceT) * 1,
      0,
      NULL,
      NULL,
      &status);

  ceT val = to_cl_type(eT(0));

  if ((status == CL_SUCCESS) && (mapped_ptr != NULL))
    {
    val = *((ceT*) (mapped_ptr));

    status = coot_wrapper(clEnqueueUnmapMemObject)(
        get_rt().cl_rt.get_cq(),
        mem.cl_mem_ptr.ptr,
        mapped_ptr,
        0,
        NULL,
        NULL);
    }

  coot_check_cl_error(status, "opencl::get_val(): couldn't access device memory");

  return from_cl_type<eT, ceT>(val);

  // coot_aligned eT               val;
  // coot_aligned cl_buffer_region region;
  //
  // region.origin = sizeof(eT)*ii;~
  // region.size   = sizeof(eT)*1;
  //
  // cl_int status = 0;
  // // NOTE: the origin must be a multiple of alignment: see coot_cl_rt_meat how to get the alignment
  // cl_mem sub_mem = clCreateSubBuffer(dev_mem, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
  //
  // coot_check_runtime_error( (status != CL_SUCCESS), "Mat::operator(): couldn't create subbuffer" );
  //
  // status |= clEnqueueReadBuffer(get_rt().cl_rt.get_cq(), sub_mem, CL_TRUE, 0, sizeof(eT)*1, &val, 0, NULL, NULL);
  // status |= clFinish(get_rt().cl_rt.get_cq());
  //
  // coot_check_runtime_error( (status != CL_SUCCESS), "Mat::operator(): couldn't read from device memory" );
  //
  // return val;
  }



template<typename eT>
inline
void
set_val(dev_mem_t<eT> mem, const uword index, const eT in_val)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(get_rt().cl_rt.get_cq(),
                                                                   mem.cl_mem_ptr.ptr,
                                                                   CL_TRUE,
                                                                   CL_MAP_WRITE,
                                                                   sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
                                                                   sizeof(ceT) * 1,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   &status);

  if( (status == CL_SUCCESS) && (mapped_ptr != NULL) )
    {
    ceT cl_val = to_cl_type(in_val);
    *((ceT*)(mapped_ptr)) = cl_val;

    status = coot_wrapper(clEnqueueUnmapMemObject)(get_rt().cl_rt.get_cq(), mem.cl_mem_ptr.ptr, mapped_ptr, 0, NULL, NULL);
    }

  coot_check_cl_error(status, "opencl::set_val(): couldn't access device memory" );
  }



template<typename eT>
inline
void
val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(get_rt().cl_rt.get_cq(),
                                                                   mem.cl_mem_ptr.ptr,
                                                                   CL_TRUE,
                                                                   (CL_MAP_READ | CL_MAP_WRITE),
                                                                   sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
                                                                   sizeof(ceT) * 1,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   &status);

  if( (status == CL_SUCCESS) && (mapped_ptr != NULL) )
    {
    eT cl_val = from_cl_type<eT, ceT>(*((ceT*)(mapped_ptr))) + val;
    *((ceT*)(mapped_ptr)) = to_cl_type(cl_val);

    status = coot_wrapper(clEnqueueUnmapMemObject)(get_rt().cl_rt.get_cq(), mem.cl_mem_ptr.ptr, mapped_ptr, 0, NULL, NULL);
    }

  coot_check_cl_error(status, "opencl::val_add_inplace(): couldn't access device memory" );
  }



template<typename eT>
inline
void
val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(get_rt().cl_rt.get_cq(),
                                                                   mem.cl_mem_ptr.ptr,
                                                                   CL_TRUE,
                                                                   (CL_MAP_READ | CL_MAP_WRITE),
                                                                   sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
                                                                   sizeof(ceT) * 1,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   &status);

  if( (status == CL_SUCCESS) && (mapped_ptr != NULL) )
    {
    eT cl_val = from_cl_type<eT, ceT>(*((ceT*)(mapped_ptr))) - val;
    *((ceT*)(mapped_ptr)) = to_cl_type(cl_val);

    status = coot_wrapper(clEnqueueUnmapMemObject)(get_rt().cl_rt.get_cq(), mem.cl_mem_ptr.ptr, mapped_ptr, 0, NULL, NULL);
    }

  coot_check_cl_error(status, "opencl::val_minus_inplace(): couldn't access device memory" );
  }



template<typename eT>
inline
void
val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(get_rt().cl_rt.get_cq(),
                                                                   mem.cl_mem_ptr.ptr,
                                                                   CL_TRUE,
                                                                   (CL_MAP_READ | CL_MAP_WRITE),
                                                                   sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
                                                                   sizeof(ceT) * 1,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   &status);

  if( (status == CL_SUCCESS) && (mapped_ptr != NULL) )
    {
    eT cl_val = from_cl_type<eT, ceT>(*((ceT*)(mapped_ptr))) * val;
    *((ceT*)(mapped_ptr)) = to_cl_type(cl_val);

    status = coot_wrapper(clEnqueueUnmapMemObject)(get_rt().cl_rt.get_cq(), mem.cl_mem_ptr.ptr, mapped_ptr, 0, NULL, NULL);
    }

  coot_check_cl_error(status, "opencl::val_mul_inplace(): couldn't access device memory" );
  }



template<typename eT>
inline
void
val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_aligned cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;

  coot_aligned void* mapped_ptr = coot_wrapper(clEnqueueMapBuffer)(get_rt().cl_rt.get_cq(),
                                                                   mem.cl_mem_ptr.ptr,
                                                                   CL_TRUE,
                                                                   (CL_MAP_READ | CL_MAP_WRITE),
                                                                   sizeof(ceT) * (index + mem.cl_mem_ptr.offset),
                                                                   sizeof(ceT) * 1,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   &status);

  if( (status == CL_SUCCESS) && (mapped_ptr != NULL) )
    {
    eT cl_val = from_cl_type<eT, ceT>(*((ceT*)(mapped_ptr))) / val;
    *((ceT*)(mapped_ptr)) = to_cl_type(cl_val);

    status = coot_wrapper(clEnqueueUnmapMemObject)(get_rt().cl_rt.get_cq(), mem.cl_mem_ptr.ptr, mapped_ptr, 0, NULL, NULL);
    }

  coot_check_cl_error(status, "opencl::val_div_inplace(): couldn't access device memory" );
  }
