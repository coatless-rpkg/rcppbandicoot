// Copyright 2023 Ryan Curtin (http://ratml.org)
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
 * Clamp `dest` to have the elements of `src` limited to the range `[min_val, max_val]`.
 */
template<typename eT1, typename eT2>
inline
void
clamp(dev_mem_t<eT2> dest,
      const dev_mem_t<eT1> src,
      const eT1 min_val,
      const eT1 max_val,
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

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::clamp(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::clamp);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),        &(dest.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size,   cl_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),        &(src.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_src_offset.size,    cl_src_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(eT1),           &min_val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(eT1),           &max_val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_n_rows.size,        cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_n_cols.size,        cl_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 8, cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 9, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr);
  coot_check_runtime_error( (status != 0), "coot::opencl::clamp(): couldn't set input arguments");

  size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };
  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::clamp(): couldn't execute kernel");
  }
