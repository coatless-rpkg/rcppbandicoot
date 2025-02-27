// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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
 * Generate a vector with num elements; the values of the elements are linearly spaced from start to (and including) end via OpenCL.
 */
template<typename eT1, typename eT2>
inline
void
extract_cx(dev_mem_t<eT1> out_mem,
           const uword out_row_offset,
           const uword out_col_offset,
           const uword out_M_n_rows,
           const dev_mem_t<eT2> in_mem,
           const uword in_row_offset,
           const uword in_col_offset,
           const uword in_M_n_rows,
           const uword n_rows,
           const uword n_cols,
           const bool imag)
  {
  coot_extra_debug_sigprint();

  // sanity check
  static_assert( is_cx<eT2>::yes, "eT2 must be complex" );

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "opencl::extract_cx(): OpenCL runtime not valid");

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT1>(oneway_real_kernel_id::extract_cx);

  const uword out_offset = out_row_offset + out_col_offset * out_M_n_rows;
  const uword  in_offset =  in_row_offset +  in_col_offset *  in_M_n_rows;

  const uword real_or_imag = (imag) ? 1 : 0;

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_out_offset(out_offset);
  runtime_t::adapt_uword cl_in_offset(in_offset);
  runtime_t::adapt_uword cl_in_M_n_rows(in_M_n_rows);
  runtime_t::adapt_uword cl_out_M_n_rows(out_M_n_rows);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_real_or_imag(real_or_imag);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),       &in_mem.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_in_offset.size,    cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),       &out_mem.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_out_offset.size,   cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_real_or_imag.size, cl_real_or_imag.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_n_rows.size,       cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_n_cols.size,       cl_n_cols.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_in_M_n_rows.size,  cl_in_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 8, cl_out_M_n_rows.size, cl_out_M_n_rows.addr);

  const size_t global_work_size[2] = { n_rows, n_cols };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::extract_cx(): couldn't execute kernel");
  }
