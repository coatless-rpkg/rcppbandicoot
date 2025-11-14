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
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::yes,
  void
  >::result
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

  const size_t buffer_origin[3] = { sizeof(eT) * src_row_offset + src.cl_mem_ptr.offset, src_col_offset, 0 };
  const size_t host_origin[3]   = { 0,                                                   0,              0 };
  const size_t region[3]        = { sizeof(eT) * n_rows,                                 n_cols,         1 };
  // use a blocking call
  const cl_int status = coot_wrapper(clEnqueueReadBufferRect)(get_rt().cl_rt.get_cq(),
                                                              src.cl_mem_ptr.ptr,
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
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::no,
  void
  >::result
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  typedef typename cl_type<eT>::type ceT;
  ceT* tmp_mem = cpu_memory::acquire<ceT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  const size_t buffer_origin[3] = { sizeof(ceT) * src_row_offset + src.cl_mem_ptr.offset, src_col_offset, 0 };
  const size_t host_origin[3]   = { 0,                                                    0,              0 };
  const size_t region[3]        = { sizeof(ceT) * n_rows,                                 n_cols,         1 };
  // use a blocking call
  const cl_int status = coot_wrapper(clEnqueueReadBufferRect)(get_rt().cl_rt.get_cq(),
                                                              src.cl_mem_ptr.ptr,
                                                              CL_TRUE,
                                                              buffer_origin,
                                                              host_origin,
                                                              region,
                                                              sizeof(ceT) * src_M_n_rows,
                                                              0,
                                                              sizeof(ceT) * n_rows,
                                                              0,
                                                              tmp_mem,
                                                              0,
                                                              NULL,
                                                              NULL);

  if (status == 0)
    {
    for (uword i = 0; i < n_rows * n_cols; ++i)
      {
      dest[i] = from_cl_type<eT, ceT>(tmp_mem[i]);
      }
    }

  cpu_memory::release(tmp_mem);

  coot_check_cl_error(status, "Mat::copy_from_dev_mem(): couldn't access device memory" );

  }



template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::yes,
  void
  >::result
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = coot_wrapper(clEnqueueWriteBuffer)(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr.ptr, CL_TRUE, dest.cl_mem_ptr.offset, sizeof(eT) * N, src, 0, NULL, NULL);

  coot_check_cl_error(status, "Mat::write_dev_mem(): couldn't access device memory");
  }



// Tediously convert all host values to the correct device value.
template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::no,
  void
  >::result
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  typedef typename cl_type<eT>::type ceT;
  ceT* tmp_mem = cpu_memory::acquire<ceT>(N);

  for (uword i = 0; i < N; ++i)
    {
    tmp_mem[i] = to_cl_type(src[i]);
    }

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = coot_wrapper(clEnqueueWriteBuffer)(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr.ptr, CL_TRUE, dest.cl_mem_ptr.offset, sizeof(ceT) * N, tmp_mem, 0, NULL, NULL);

  cpu_memory::release(tmp_mem);

  coot_check_cl_error(status, "Mat::write_dev_mem(): couldn't access device memory");
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

  const uword dest_offset = dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src.cl_mem_ptr.offset +  src_row_offset +  src_col_offset * src_M_n_rows;

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),        &(dest.cl_mem_ptr.ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size,   cl_dest_offset.addr      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),        &( src.cl_mem_ptr.ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_src_offset.size,    cl_src_offset.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_n_rows.size,        cl_n_rows.addr           );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_n_cols.size,        cl_n_cols.addr           );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr     );

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_mat(): couldn't copy buffer");
  }



template<typename eT2, typename eT1>
inline
void
copy_cube(dev_mem_t<eT2> dest,
          const dev_mem_t<eT1> src,
          // logical size of cube
          const uword n_rows,
          const uword n_cols,
          const uword n_slices,
          // offsets for subviews
          const uword dest_row_offset,
          const uword dest_col_offset,
          const uword dest_slice_offset,
          const uword dest_M_n_rows,
          const uword dest_M_n_cols,
          const uword src_row_offset,
          const uword src_col_offset,
          const uword src_slice_offset,
          const uword src_M_n_rows,
          const uword src_M_n_cols)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0 || n_slices == 0)
    {
    return;
    }

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::convert_type_cube);

  const uword dest_offset = dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows + dest_slice_offset * dest_M_n_rows * dest_M_n_cols;
  const uword  src_offset =  src.cl_mem_ptr.offset +  src_row_offset +  src_col_offset * src_M_n_rows  +  src_slice_offset * src_M_n_rows * src_M_n_cols;;

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_n_slices(n_slices);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_dest_M_n_cols(dest_M_n_cols);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);
  runtime_t::adapt_uword cl_src_M_n_cols(src_M_n_cols);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0, sizeof(cl_mem),        &(dest.cl_mem_ptr.ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  1, cl_dest_offset.size,   cl_dest_offset.addr      );
  status |= coot_wrapper(clSetKernelArg)(kernel,  2, sizeof(cl_mem),        &( src.cl_mem_ptr.ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  3, cl_src_offset.size,    cl_src_offset.addr       );
  // these two arguments are ignored
  status |= coot_wrapper(clSetKernelArg)(kernel,  4, sizeof(cl_mem),        &( src.cl_mem_ptr.ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  5, cl_src_offset.size,    cl_src_offset.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel,  6, cl_n_rows.size,        cl_n_rows.addr           );
  status |= coot_wrapper(clSetKernelArg)(kernel,  7, cl_n_cols.size,        cl_n_cols.addr           );
  status |= coot_wrapper(clSetKernelArg)(kernel,  8, cl_n_slices.size,      cl_n_slices.addr         );
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_dest_M_n_cols.size, cl_dest_M_n_cols.addr    );
  // these two arguments are ignored
  status |= coot_wrapper(clSetKernelArg)(kernel, 11, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 12, cl_src_M_n_cols.size,  cl_src_M_n_cols.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 13, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 14, cl_src_M_n_cols.size,  cl_src_M_n_cols.addr     );

  const size_t global_work_size[3] = { size_t(n_rows), size_t(n_cols), size_t(n_slices) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_cube(): couldn't copy buffer");
  }
