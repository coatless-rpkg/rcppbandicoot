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



template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::yes,
  void
  >::result
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  runtime_t::cq_guard guard;

  const size_t buffer_origin[3] = { sizeof(eT) * src_row_offset + src.cl_mem_ptr.offset, src_col_offset, 0 };
  const size_t host_origin[3]   = { 0,                                                   0,              0 };
  const size_t region[3]        = { sizeof(eT) * n_rows,                                 n_cols,         1 };
  // use a blocking call
  const cl_int status = coot_wrapper(clEnqueueReadBufferRect)(get_rt().cl_rt.get_cq(),
                                                              src.cl_mem_ptr.ptr,
                                                              CL_TRUE,
                                                              buffer_origin,
                                                              host_origin,
                                                              region,
                                                              sizeof(eT) * src_M_n_rows,
                                                              0,
                                                              sizeof(eT) * n_rows,
                                                              0,
                                                              dest,
                                                              0,
                                                              NULL,
                                                              NULL);

  coot_check_cl_error(status, "Mat::copy_from_dev_mem(): couldn't access device memory" );
  }



template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::no,
  void
  >::result
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  typedef typename cl_type<eT>::type ceT;
  cpu_memory::mem_array<ceT> tmp_mem_array(n_rows * n_cols);
  ceT* tmp_mem = tmp_mem_array.memptr();

  runtime_t::cq_guard guard;

  const size_t buffer_origin[3] = { sizeof(ceT) * src_row_offset + src.cl_mem_ptr.offset, src_col_offset, 0 };
  const size_t host_origin[3]   = { 0,                                                    0,              0 };
  const size_t region[3]        = { sizeof(ceT) * n_rows,                                 n_cols,         1 };
  // use a blocking call
  const cl_int status = coot_wrapper(clEnqueueReadBufferRect)(get_rt().cl_rt.get_cq(),
                                                              src.cl_mem_ptr.ptr,
                                                              CL_TRUE,
                                                              buffer_origin,
                                                              host_origin,
                                                              region,
                                                              sizeof(ceT) * src_M_n_rows,
                                                              0,
                                                              sizeof(ceT) * n_rows,
                                                              0,
                                                              tmp_mem,
                                                              0,
                                                              NULL,
                                                              NULL);

  if (status == 0)
    {
    for (uword i = 0; i < n_rows * n_cols; ++i)
      {
      dest[i] = from_cl_type<eT, ceT>(tmp_mem[i]);
      }
    }

  coot_check_cl_error(status, "Mat::copy_from_dev_mem(): couldn't access device memory" );

  }



template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::yes,
  void
  >::result
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_debug_sigprint();

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = coot_wrapper(clEnqueueWriteBuffer)(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr.ptr, CL_TRUE, dest.cl_mem_ptr.offset, sizeof(eT) * N, src, 0, NULL, NULL);

  coot_check_cl_error(status, "Mat::write_dev_mem(): couldn't access device memory");
  }



// Tediously convert all host values to the correct device value.
template<typename eT>
inline
typename
enable_if2
  <
  is_same_type< eT, typename cl_type<eT>::type >::no,
  void
  >::result
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_debug_sigprint();

  typedef typename cl_type<eT>::type ceT;
  cpu_memory::mem_array<ceT> tmp_mem_array(N);
  ceT* tmp_mem = tmp_mem_array.memptr();

  for (uword i = 0; i < N; ++i)
    {
    tmp_mem[i] = to_cl_type(src[i]);
    }

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = coot_wrapper(clEnqueueWriteBuffer)(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr.ptr, CL_TRUE, dest.cl_mem_ptr.offset, sizeof(ceT) * N, tmp_mem, 0, NULL, NULL);

  coot_check_cl_error(status, "Mat::write_dev_mem(): couldn't access device memory");
  }



template<typename T1, typename T2>
inline
void
copy(const Proxy<T1>& out, const Proxy<T2>& in)
  {
  coot_debug_sigprint();

  if (in.is_empty())
    {
    return;
    }

  // this should never happen since coot_rt_t casts T2 to the right dimensions
  coot_static_check( Proxy<T1>::num_dims != Proxy<T2>::num_dims, "coot::cuda::copy(): objects must have the same number of dimensions" );

  cl_kernel kernel = get_rt().cl_rt.get_kernel<kernel_id::copy, Proxy<T1>, Proxy<T2>>();

  // We need to instantiate all arguments with the right type in order to run the kernel.
  typename cl_args<Proxy<T1>, Proxy<T2>>::result args = to_cl_args(out, in);

  // Now set all the arguments.
  set_args(kernel, "coot::opencl::copy()", args);

  std::array<size_t, Proxy<T1>::num_dims> work_size = get_work_size(out);

  const cl_int status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, Proxy<T1>::num_dims, NULL, work_size.data(), NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy(): couldn't execute kernel");
  }
