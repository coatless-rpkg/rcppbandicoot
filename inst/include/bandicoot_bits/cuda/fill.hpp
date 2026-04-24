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

  CUfunction kernel = get_rt().cuda_rt.get_kernel<kernel_id::fill, Proxy<T1>>();

  typedef typename cuda_type<typename Proxy<T1>::elem_type>::type ceT;
  const ceT conv_val = to_cuda_type(val);

  const auto& args = construct_args(dest, val);
  const kernel_dims dims = grid_dims(dest);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args.data(),
      0);

  coot_check_cuda_error( result, "coot::cuda::fill(): cuLaunchKernel() failed" );
  }
