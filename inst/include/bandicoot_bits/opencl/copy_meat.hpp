// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  const size_t buffer_origin[3] = { sizeof(eT) * src_row_offset, src_col_offset, 0 };
  const size_t host_origin[3]   = { 0,                           0,              0 };
  const size_t region[3]        = { sizeof(eT) * n_rows,         n_cols,         1 };
  // use a blocking call
  const cl_int status = coot_wrapper(clEnqueueReadBufferRect)(get_rt().cl_rt.get_cq(),
                                                              src.cl_mem_ptr,
                                                              CL_TRUE,
                                                              buffer_origin,
                                                              host_origin,
                                                              region,
                                                              sizeof(eT) * src_M_n_rows,
                                                              0,
                                                              sizeof(eT) * n_rows,
                                                              0,
                                                              dest,
                                                              0,
                                                              NULL,
                                                              NULL);

  coot_check_cl_error(status, "Mat::copy_from_dev_mem(): couldn't access device memory" );
  }



template<typename eT>
inline
void
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = coot_wrapper(clEnqueueWriteBuffer)(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr, CL_TRUE, 0, sizeof(eT)*N, src, 0, NULL, NULL);

  coot_check_cl_error(status, "Mat::write_dev_mem(): couldn't access device memory" );
  }



/**
 * Copy an array via OpenCL and cast the type of its elements.
 */
template<typename eT2, typename eT1>
inline
void
copy_mat(dev_mem_t<eT2> dest,
         const dev_mem_t<eT1> src,
         const uword n_rows,
         const uword n_cols,
         const uword dest_row_offset,
         const uword dest_col_offset,
         const uword dest_M_n_rows,
         const uword src_row_offset,
         const uword src_col_offset,
         const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::convert_type);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),        &(dest.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size,   cl_dest_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),        &( src.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_src_offset.size,    cl_src_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_n_rows.size,        cl_n_rows.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_n_cols.size,        cl_n_cols.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr );

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_mat(): couldn't copy buffer");
  }
