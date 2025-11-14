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
 * Run an OpenCL elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
fill(dev_mem_t<eT> dest,
     const eT val,
     const uword n_rows,
     const uword n_cols,
     const uword row_offset,
     const uword col_offset,
     const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    return;

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::fill);

  runtime_t::cq_guard guard;

  const uword out_offset = dest.cl_mem_ptr.offset + row_offset + col_offset * M_n_rows;
  runtime_t::adapt_uword cl_out_offset(out_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_M_n_rows(M_n_rows);

  cl_int status = 0;

  typedef typename cl_type<eT>::type ceT;
  ceT cl_val = to_cl_type(val);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),     &(dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_out_offset.size, cl_out_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(ceT),        &cl_val               );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_n_rows.size,     cl_n_rows.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_n_cols.size,     cl_n_cols.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_M_n_rows.size,   cl_M_n_rows.addr      );
  coot_check_cl_error(status, "coot::opencl::fill(): couldn't set kernel arguments");

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::fill(): couldn't execute kernel");
  }
