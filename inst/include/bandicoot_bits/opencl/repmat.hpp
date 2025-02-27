// Copyright 2022 Ryan Curtin (http://www.ratml.org)
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
 * Copy the given matrix `src` into `dest`, making `copies_per_col` copies of each column, and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
repmat(const dev_mem_t<eT1> src, dev_mem_t<eT2> dest, const uword n_rows, const uword n_cols, const uword copies_per_row, const uword copies_per_col)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::repmat(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  const uword new_n_rows = n_rows * copies_per_row;

  runtime_t::adapt_uword local_n_rows(n_rows);
  runtime_t::adapt_uword local_n_cols(n_cols);
  runtime_t::adapt_uword local_copies_per_row(copies_per_row);
  runtime_t::adapt_uword local_copies_per_col(copies_per_col);
  runtime_t::adapt_uword local_new_n_rows(new_n_rows);
  runtime_t::adapt_uword local_src_offset(src.cl_mem_ptr.offset);
  runtime_t::adapt_uword local_dest_offset(dest.cl_mem_ptr.offset);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::repmat);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),            &(src.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, local_src_offset.size,     local_src_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),            &(dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, local_dest_offset.size,    local_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, local_n_rows.size,         local_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, local_n_cols.size,         local_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, local_copies_per_row.size, local_copies_per_row.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, local_copies_per_col.size, local_copies_per_col.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 8, local_new_n_rows.size,     local_new_n_rows.addr);

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::repmat(): couldn't execute kernel" );
  }
