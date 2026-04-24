// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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



template<size_t i, typename T>
inline
void
set_arg(const cl_kernel& kernel, const std::string& func_name, const T& t)
  {
  const cl_int status = coot_wrapper(clSetKernelArg)(kernel, i, sizeof(t), &t);
  coot_check_cl_error(status, func_name + ": failed to set kernel argument " + std::to_string(i));
  }



template<size_t i>
inline
void
set_arg(const cl_kernel& kernel, const std::string& func_name, const runtime_t::adapt_uword& t)
  {
  const cl_int status = coot_wrapper(clSetKernelArg)(kernel, i, t.size, t.addr);
  coot_check_cl_error(status, func_name + ": failed to set kernel argument " + std::to_string(i));
  }



template<size_t offset, typename... Ts>
inline
typename
enable_if2
  <
  offset == sizeof...(Ts) - 1,
  void
  >::result
set_args_inner(const cl_kernel& kernel, const std::string& func_name, const std::tuple< Ts... >& args)
  {
  set_arg<offset>(kernel, func_name, std::get<offset>(args));
  }



template<size_t offset, typename... Ts>
inline
typename
enable_if2
  <
  offset + 1 < sizeof...(Ts),
  void
  >::result
set_args_inner(const cl_kernel& kernel, const std::string& func_name, const std::tuple< Ts... >& args)
  {
  set_arg<offset>(kernel, func_name, std::get<offset>(args));
  set_args_inner<offset + 1>(kernel, func_name, args);
  }



template<typename T, typename... Ts>
inline
void
set_args(const cl_kernel& kernel, const std::string& func_name, const std::tuple< T, Ts... >& args)
  {
  set_args_inner<0>(kernel, func_name, args);
  }
