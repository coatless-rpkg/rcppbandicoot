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



template<typename eT1, typename eT2>
inline
void
relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);
  runtime_t::adapt_uword out_mem_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword in_mem_offset(in_mem.cl_mem_ptr.offset);

  typedef typename cl_type<eT2>::type ceT2;
  ceT2 cl_val = to_cl_type(val);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &out_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, out_mem_offset.size, out_mem_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &in_mem.cl_mem_ptr.ptr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, in_mem_offset.size,  in_mem_offset.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, N.size,              N.addr                 );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(ceT2),        &cl_val                );
  coot_check_cl_error(status, "coot::opencl::relational_scalar_op() (" + name + "): couldn't set kernel arguments");

  size_t work_size = size_t(n_elem);

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::relational_scalar_op() (" + name + "): couldn't execute kernel");
  }



template<typename eT1>
inline
void
relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT1>(num);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);
  runtime_t::adapt_uword out_mem_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword in_mem_offset(in_mem.cl_mem_ptr.offset);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &out_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, out_mem_offset.size, out_mem_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &in_mem.cl_mem_ptr.ptr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, in_mem_offset.size,  in_mem_offset.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, N.size,              N.addr                 );

  coot_check_cl_error(status, "coot::opencl::relational_unary_array_op() (" + name + "): couldn't set kernel arguments");

  size_t work_size = size_t(n_elem);

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::relational_unary_array_op() (" + name + "): couldn't execute kernel");
  }



template<typename eT1, typename eT2>
inline
void
relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);
  runtime_t::adapt_uword out_mem_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword X_mem_offset(X_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword Y_mem_offset(Y_mem.cl_mem_ptr.offset);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &out_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, out_mem_offset.size, out_mem_offset.addr    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &X_mem.cl_mem_ptr.ptr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, X_mem_offset.size,   X_mem_offset.addr      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),      &Y_mem.cl_mem_ptr.ptr  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, Y_mem_offset.size,   Y_mem_offset.addr      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, N.size,              N.addr                 );
  coot_check_cl_error(status, "coot::opencl::relational_array_op() (" + name + "): couldn't set kernel arguments");

  size_t work_size = size_t(n_elem);

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::relational_array_op() (" + name + "): couldn't execute kernel");
  }
