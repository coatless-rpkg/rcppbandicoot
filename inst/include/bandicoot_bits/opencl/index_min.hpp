// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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



template<typename eT1>
inline
void
index_min(dev_mem_t<uword> dest,
          const dev_mem_t<eT1> src,
          const uword n_rows,
          const uword n_cols,
          const uword dim,
          // subview arguments
          const uword dest_offset,
          const uword dest_mem_incr,
          const uword src_row_offset,
          const uword src_col_offset,
          const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::index_min(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel kernel;
  if (dim == 0)
    {
    kernel = get_rt().cl_rt.get_kernel<eT1>(oneway_kernel_id::index_min_colwise);
    }
  else
    {
    kernel = get_rt().cl_rt.get_kernel<eT1>(oneway_kernel_id::index_min_rowwise);
    }

  cl_int status = 0;

  const uword src_offset = src.cl_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows;

  runtime_t::adapt_uword cl_dest_offset(dest_offset + dest.cl_mem_ptr.offset);
  runtime_t::adapt_uword cl_src_offset(src_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_mem_incr(dest_mem_incr);
  runtime_t::adapt_uword cl_src_M_n_rows(src_M_n_rows);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),        &(dest.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size,   cl_dest_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),        &(src.cl_mem_ptr.ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_src_offset.size,    cl_src_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_n_rows.size,        cl_n_rows.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_n_cols.size,        cl_n_cols.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_dest_mem_incr.size, cl_dest_mem_incr.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, cl_src_M_n_rows.size,  cl_src_M_n_rows.addr  );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0                                    };
  const size_t k1_work_size[1]   = { size_t((dim == 0) ? n_cols : n_rows) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::index_min(): failed to run kernel");
  }



/**
 * Compute the index of the minimum element in `mem`.
 * This is basically the same as accu(), which is also a reduction.
 */
template<typename eT>
inline
uword
index_min_vec(dev_mem_t<eT> mem, const uword n_elem, eT* min_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::index_min_vec(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::index_min);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::index_min_small);

  return generic_reduce_uword_aux<eT, eT>(mem, n_elem, "index_min_vec", min_val, k, k_small, std::make_tuple(/* no extra args */));
  }
