// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
 * Replace `val_find` with `val_replace`.
 */
template<typename eT1, typename eT2>
inline
void
replace(dev_mem_t<eT2> dest,
        const dev_mem_t<eT1> src,
        const eT1 val_find,
        const eT1 val_replace,
        const uword n_rows,
        const uword n_cols,
        const uword n_slices,
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

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::replace);

  const uword src_offset = src.cl_mem_ptr.offset   + src_row_offset  + src_col_offset * src_M_n_rows   + src_slice_offset * src_M_n_rows * src_M_n_cols;
  const uword dest_offset = dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows + dest_slice_offset * dest_M_n_rows * dest_M_n_cols;

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_n_slices(n_slices);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);
  runtime_t::adapt_uword cl_src_M_n_cols(src_M_n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_dest_M_n_cols(dest_M_n_cols);


  typedef typename cl_type<eT1>::type ceT1;
  ceT1 cl_val_find = to_cl_type(val_find);
  ceT1 cl_val_replace = to_cl_type(val_replace);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0,  sizeof(cl_mem),        &dest.cl_mem_ptr.ptr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1,  cl_dest_offset.size,   cl_dest_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2,  sizeof(cl_mem),        &src.cl_mem_ptr.ptr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3,  cl_src_offset.size,    cl_src_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4,  sizeof(ceT1),          &cl_val_find         );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5,  sizeof(ceT1),          &cl_val_replace      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6,  cl_n_rows.size,        cl_n_rows.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 7,  cl_n_cols.size,        cl_n_cols.addr       );
  status |= coot_wrapper(clSetKernelArg)(kernel, 8,  cl_n_slices.size,      cl_n_slices.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 9,  cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_dest_M_n_cols.size, cl_dest_M_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 11, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 12, cl_src_M_n_cols.size,  cl_src_M_n_cols.addr );

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::replace(): couldn't set kernel arguments");

  size_t work_size[3] = { size_t(n_rows), size_t(n_cols), size_t(n_slices) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 3, NULL, work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::replace(): couldn't execute kernel");
  }
