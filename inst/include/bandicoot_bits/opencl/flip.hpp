// Copyright 2026 Andrew Furey (http://andrew.industries)
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
fliplr(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols, const uword src_M_n_rows, const uword aux_row1, const uword aux_col1)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cl_rt.is_valid() == false), "coot::opencl::fliplr(): OpenCL runtime not valid");

  runtime_t::cq_guard guard;

  const uword subview_offset = aux_row1 + src_M_n_rows * aux_col1;

  runtime_t::adapt_uword local_n_rows(n_rows);
  runtime_t::adapt_uword local_n_cols(n_cols);
  runtime_t::adapt_uword local_in_offset(in.cl_mem_ptr.offset + subview_offset);
  runtime_t::adapt_uword local_out_offset(out.cl_mem_ptr.offset);
  runtime_t::adapt_uword local_src_M_n_rows(src_M_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::fliplr);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),          &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, local_out_offset.size,   local_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),          &(in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, local_in_offset.size,    local_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, local_n_rows.size,       local_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, local_n_cols.size,       local_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, local_src_M_n_rows.size, local_src_M_n_rows.addr);

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::fliplr(): couldn't execute kernel" );
  }



template<typename eT>
inline
void
flipud(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols, const uword src_M_n_rows, const uword aux_row1, const uword aux_col1)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cl_rt.is_valid() == false), "coot::opencl::flipud(): OpenCL runtime not valid");

  runtime_t::cq_guard guard;

  const uword subview_offset = aux_row1 + src_M_n_rows * aux_col1;

  runtime_t::adapt_uword local_n_rows(n_rows);
  runtime_t::adapt_uword local_n_cols(n_cols);
  runtime_t::adapt_uword local_in_offset(in.cl_mem_ptr.offset + subview_offset);
  runtime_t::adapt_uword local_out_offset(out.cl_mem_ptr.offset);
  runtime_t::adapt_uword local_src_M_n_rows(src_M_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::flipud);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),          &(out.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, local_out_offset.size,   local_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),          &(in.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, local_in_offset.size,    local_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, local_n_rows.size,       local_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, local_n_cols.size,       local_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, local_src_M_n_rows.size, local_src_M_n_rows.addr);

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::flipud(): couldn't execute kernel" );
  }
