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
 * Accumulate all elements in `mem`.
 */
template<typename eT>
inline
eT
accu(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::accu(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  return generic_reduce<eT, eT>(mem, n_elem, "accu", k, k_small, std::make_tuple(/* no extra args */));
  }



template<typename eT>
inline
eT
accu_subview(dev_mem_t<eT> mem, const uword m_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::accu(): OpenCL runtime not valid" );

  // TODO: implement specialised handling for two cases: (i) n_cols = 1, (ii) n_rows = 1

  Mat<eT> tmp(1, n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k1 = get_rt().cl_rt.get_kernel<eT, eT>(twoway_kernel_id::sum_colwise_conv_pre);

  cl_int status = 0;

  const uword dest_offset = 0;
  const uword  src_offset = aux_row1 + aux_col1 * m_n_rows;
  const uword dest_mem_incr = 1;

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_mem_incr(dest_mem_incr);
  runtime_t::adapt_uword cl_src_m_n_rows(m_n_rows);

  status |= coot_wrapper(clSetKernelArg)(k1, 0, sizeof(cl_mem),        &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k1, 1, cl_dest_offset.size,   cl_dest_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(k1, 2, sizeof(cl_mem),        &(mem.cl_mem_ptr)    );
  status |= coot_wrapper(clSetKernelArg)(k1, 3, cl_src_offset.size,    cl_src_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(k1, 4, cl_n_rows.size,        cl_n_rows.addr       );
  status |= coot_wrapper(clSetKernelArg)(k1, 5, cl_n_cols.size,        cl_n_cols.addr       );
  status |= coot_wrapper(clSetKernelArg)(k1, 6, cl_dest_mem_incr.size, cl_dest_mem_incr.addr);
  status |= coot_wrapper(clSetKernelArg)(k1, 7, cl_src_m_n_rows.size,  cl_src_m_n_rows.addr );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0              };
  const size_t k1_work_size[1]   = { size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k1, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "accu()");

  // combine the column sums

  cl_kernel k2 = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_simple);

  status |= coot_wrapper(clSetKernelArg)(k2, 0, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k2, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k2, 2, cl_n_cols.size, cl_n_cols.addr       );

  const size_t k2_work_dim       = 1;
  const size_t k2_work_offset[1] = { 0 };
  const size_t k2_work_size[1]   = { 1 };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k2, k2_work_dim, k2_work_offset, k2_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "accu()");

  return tmp(0);
  }
