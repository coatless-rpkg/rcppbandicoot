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



/**
 * Run an OpenCL non-inplace elementwise kernel.
 */
template<typename eT1, typename eT2>
inline
void
eop_scalar(const twoway_kernel_id::enum_id num,
           dev_mem_t<eT2> dest,
           const dev_mem_t<eT1> src,
           const eT1 aux_val_pre,
           const eT2 aux_val_post,
           // logical size of source and destination
           const uword n_rows,
           const uword n_cols,
           // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
           const uword dest_row_offset,
           const uword dest_col_offset,
           const uword dest_M_n_rows,
           // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
           const uword src_row_offset,
           const uword src_col_offset,
           const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  const uword src_offset = src.cl_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows;
  const uword dest_offset = dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows;

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),        &dest.cl_mem_ptr.ptr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size,   cl_dest_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),        &src.cl_mem_ptr.ptr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_src_offset.size,    cl_src_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(eT1),           &aux_val_pre         );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(eT2),           &aux_val_post        );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_n_rows.size,        cl_n_rows.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_n_cols.size,        cl_n_cols.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 8, cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 9, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr );

  size_t work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::eop_scalar(): couldn't execute kernel");
  }



/**
 * Run an OpenCL elementwise kernel that performs an operation on two matrices.
 */
template<typename eT1, typename eT2, typename eT3>
inline
void
eop_mat(const threeway_kernel_id::enum_id num,
        dev_mem_t<eT3> dest,
        const dev_mem_t<eT1> src_A,
        const dev_mem_t<eT2> src_B,
        // logical size of source and destination
        const uword n_rows,
        const uword n_cols,
        // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
        const uword dest_row_offset,
        const uword dest_col_offset,
        const uword dest_M_n_rows,
        // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
        const uword src_A_row_offset,
        const uword src_A_col_offset,
        const uword src_A_M_n_rows,
        const uword src_B_row_offset,
        const uword src_B_col_offset,
        const uword src_B_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT3, eT2, eT1>(num);

  const uword src_A_offset = src_A.cl_mem_ptr.offset + src_A_row_offset + src_A_col_offset * src_A_M_n_rows;
  const uword src_B_offset = src_B.cl_mem_ptr.offset + src_B_row_offset + src_B_col_offset * src_B_M_n_rows;
  const uword dest_offset  =  dest.cl_mem_ptr.offset +  dest_row_offset +  dest_col_offset * dest_M_n_rows;

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_A_offset(src_A_offset);
  runtime_t::adapt_uword cl_src_B_offset(src_B_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_src_A_M_n_rows(src_A_M_n_rows);
  runtime_t::adapt_uword cl_src_B_M_n_rows(src_B_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0,         sizeof(cl_mem), &( dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  1,    cl_dest_offset.size, cl_dest_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel,  2,         sizeof(cl_mem), &(src_A.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  3,   cl_src_A_offset.size, cl_src_A_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  4,         sizeof(cl_mem), &(src_B.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  5,   cl_src_B_offset.size, cl_src_B_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  6,         cl_n_rows.size, cl_n_rows.addr         );
  status |= coot_wrapper(clSetKernelArg)(kernel,  7,         cl_n_cols.size, cl_n_cols.addr         );
  status |= coot_wrapper(clSetKernelArg)(kernel,  8,  cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, cl_src_A_M_n_rows.size, cl_src_A_M_n_rows.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_src_B_M_n_rows.size, cl_src_B_M_n_rows.addr );

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::eop_mat(): couldn't execute kernel" );
  }
