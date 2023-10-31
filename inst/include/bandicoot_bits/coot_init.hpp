// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



inline
bool
coot_init(const bool print_info = false)
  {
  coot_extra_debug_sigprint();

  return init_rt(print_info);
  }



inline
bool
coot_init(const char* backend, const bool print_info = false, const uword first_id = 0, const uword second_id = 0)
  {
  coot_extra_debug_sigprint();

  uword device_id = 0;
  uword platform_id = 0;

  std::string backend_str(backend);

  coot_backend_t backend_val = CL_BACKEND;

  if (backend_str == "opencl")
    {
    backend_val = CL_BACKEND;
    platform_id = first_id;
    device_id = second_id;
    }
  else if (backend_str == "cuda")
    {
    backend_val = CUDA_BACKEND;
    device_id = first_id;
    }
  else
    {
    throw std::runtime_error("coot_init(): unknown backend '" + backend_str + "'");
    }

  return init_rt(backend_val, platform_id, device_id, print_info);
  }
  
