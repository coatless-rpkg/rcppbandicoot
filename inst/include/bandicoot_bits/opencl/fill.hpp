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



template<typename T1>
inline
void
fill(const Proxy<T1>& dest, const typename T1::elem_type val)
  {
  coot_debug_sigprint();

  if (dest.is_empty())
    {
    return;
    }

  cl_kernel kernel = get_rt().cl_rt.get_kernel<kernel_id::fill, Proxy<T1>>();

  // We need to instantiate all arguments with the right type in order to run the kernel.
  typedef typename std::tuple_element<0, typename to_cl_types<typename T1::elem_type>::result>::type ceT;
  typename cl_args<Proxy<T1>, const ceT>::result args = to_cl_args(dest, val);

  // Now set all the arguments.
  set_args(kernel, "coot::opencl::fill()", args);

  std::array<size_t, Proxy<T1>::num_dims> work_size = get_work_size(dest);

  const cl_int status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, Proxy<T1>::num_dims, NULL, work_size.data(), NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::fill(): couldn't execute kernel");
  }
