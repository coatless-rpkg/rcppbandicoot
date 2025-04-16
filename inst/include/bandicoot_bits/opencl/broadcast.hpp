// Copyright 2022-2025 Ryan Curtin (http://www.ratml.org)
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
 * Perform a broadcasting operation from the matrix `src` into `dest`, making `copies_per_col` copies of each column
 * and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
broadcast_op(const twoway_kernel_id::enum_id op,
             dev_mem_t<eT2> dest,
             const dev_mem_t<eT2> dest_in,
             const dev_mem_t<eT1> src,
             const uword src_n_rows,
             const uword src_n_cols,
             const uword copies_per_row,
             const uword copies_per_col,
             const uword dest_row_offset,
             const uword dest_col_offset,
             const uword dest_M_n_rows,
             const uword dest_in_row_offset,
             const uword dest_in_col_offset,
             const uword dest_in_M_n_rows,
             const uword src_row_offset,
             const uword src_col_offset,
             const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::broadcast_op(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword local_src_n_rows(src_n_rows);
  runtime_t::adapt_uword local_src_n_cols(src_n_cols);
  runtime_t::adapt_uword local_copies_per_row(copies_per_row);
  runtime_t::adapt_uword local_copies_per_col(copies_per_col);
  runtime_t::adapt_uword local_src_offset(src.cl_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows);
  runtime_t::adapt_uword local_dest_offset(dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows);
  runtime_t::adapt_uword local_dest_in_offset(dest_in.cl_mem_ptr.offset + dest_in_row_offset + dest_in_col_offset * dest_in_M_n_rows);
  runtime_t::adapt_uword local_src_M_n_rows(src_M_n_rows);
  runtime_t::adapt_uword local_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword local_dest_in_M_n_rows(dest_in_M_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(op);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0, sizeof(cl_mem),              &(dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  1, local_dest_offset.size,      local_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  2, sizeof(cl_mem),              &(dest_in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  3, local_dest_in_offset.size,   local_dest_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  4, sizeof(cl_mem),              &(src.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  5, local_src_offset.size,       local_src_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  6, local_src_n_rows.size,       local_src_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  7, local_src_n_cols.size,       local_src_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  8, local_copies_per_row.size,   local_copies_per_row.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, local_copies_per_col.size,   local_copies_per_col.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, local_dest_M_n_rows.size,    local_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 11, local_dest_in_M_n_rows.size, local_dest_in_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 12, local_src_M_n_rows.size,     local_src_M_n_rows.addr);

  const size_t global_work_size[2] = { size_t(src_n_rows * copies_per_row), size_t(src_n_cols * copies_per_col) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::broadcast_op(): couldn't execute kernel" );
  }



/**
 * Perform a broadcasting operation on a subset of the rows or columns of the matrix `src` into `dest`,
 * making `copies_per_col` copies of each column and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
broadcast_subset_op(const twoway_kernel_id::enum_id op,
                    dev_mem_t<eT2> dest,
                    const dev_mem_t<eT2> dest_in,
                    const dev_mem_t<eT1> src,
                    const dev_mem_t<uword> indices,
                    const uword mode,
                    const uword src_n_rows,
                    const uword src_n_cols,
                    const uword copies_per_row,
                    const uword copies_per_col,
                    const uword dest_row_offset,
                    const uword dest_col_offset,
                    const uword dest_M_n_rows,
                    const uword dest_in_row_offset,
                    const uword dest_in_col_offset,
                    const uword dest_in_M_n_rows,
                    const uword src_row_offset,
                    const uword src_col_offset,
                    const uword src_M_n_rows,
                    const uword indices_offset,
                    const uword indices_incr)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::broadcast_subset_op(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword local_mode(mode);
  runtime_t::adapt_uword local_src_n_rows(src_n_rows);
  runtime_t::adapt_uword local_src_n_cols(src_n_cols);
  runtime_t::adapt_uword local_copies_per_row(copies_per_row);
  runtime_t::adapt_uword local_copies_per_col(copies_per_col);
  runtime_t::adapt_uword local_src_offset(src.cl_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows);
  runtime_t::adapt_uword local_dest_offset(dest.cl_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows);
  runtime_t::adapt_uword local_dest_in_offset(dest_in.cl_mem_ptr.offset + dest_in_row_offset + dest_in_col_offset * dest_in_M_n_rows);
  runtime_t::adapt_uword local_indices_offset(indices.cl_mem_ptr.offset + indices_offset);
  runtime_t::adapt_uword local_src_M_n_rows(src_M_n_rows);
  runtime_t::adapt_uword local_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword local_dest_in_M_n_rows(dest_in_M_n_rows);
  runtime_t::adapt_uword local_indices_incr(indices_incr);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(op);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0, sizeof(cl_mem),              &(dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  1, local_dest_offset.size,      local_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  2, sizeof(cl_mem),              &(dest_in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  3, local_dest_in_offset.size,   local_dest_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  4, sizeof(cl_mem),              &(src.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  5, local_src_offset.size,       local_src_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  6, sizeof(cl_mem),              &(indices.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel,  7, local_indices_offset.size,   local_indices_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  8, local_mode.size,             local_mode.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, local_src_n_rows.size,       local_src_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, local_src_n_cols.size,       local_src_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 11, local_copies_per_row.size,   local_copies_per_row.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 12, local_copies_per_col.size,   local_copies_per_col.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 13, local_dest_M_n_rows.size,    local_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 14, local_dest_in_M_n_rows.size, local_dest_in_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 15, local_src_M_n_rows.size,     local_src_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 16, local_indices_incr.size,     local_indices_incr.addr);

  const size_t global_work_size[2] = { size_t(src_n_rows * copies_per_row), size_t(src_n_cols * copies_per_col) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::broadcast_subset_op(): couldn't execute kernel" );
  }
