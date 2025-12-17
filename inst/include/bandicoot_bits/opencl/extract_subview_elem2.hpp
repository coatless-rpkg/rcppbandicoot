// Copyright 2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT1, typename eT2>
inline
void
extract_subview_elem2(dev_mem_t<eT2> out_mem,
                      const dev_mem_t<eT1> in_mem,
                      const dev_mem_t<uword> in_row_locs,
                      const dev_mem_t<uword> in_col_locs,
                      const uword n_row_elems,
                      const uword n_col_elems,
                      const uword out_n_rows,
                      const uword in_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "opencl::extract_subview_elem2(): OpenCL runtime not valid");

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::extract_sve2);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword cl_out_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_in_offset(in_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_in_row_locs_offset(in_row_locs.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_in_col_locs_offset(in_col_locs.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_n_row_elems(n_row_elems);
  runtime_t::adapt_uword cl_n_col_elems(n_col_elems);
  runtime_t::adapt_uword cl_out_n_rows(out_n_rows);
  runtime_t::adapt_uword cl_in_n_rows(in_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0, sizeof(cl_mem),             &out_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  1, cl_out_offset.size,         cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  2, sizeof(cl_mem),             &in_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  3, cl_in_offset.size,          cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  4, sizeof(cl_mem),             &in_row_locs.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  5, cl_in_row_locs_offset.size, cl_in_row_locs_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  6, sizeof(cl_mem),             &in_col_locs.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  7, cl_in_col_locs_offset.size, cl_in_col_locs_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  8, cl_n_row_elems.size,        cl_n_row_elems.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, cl_n_col_elems.size,        cl_n_col_elems.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_out_n_rows.size,         cl_out_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 11, cl_in_n_rows.size,          cl_in_n_rows.addr);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::extract_subview_elem2(): couldn't set kernel arguments");

  const size_t work_size[2] = { size_t(n_row_elems), size_t(n_col_elems) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::extract_subview_elem2(): couldn't execute kernel");
  }
