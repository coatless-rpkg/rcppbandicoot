// SPDX-License-Identifier: Apache-2.0
// 
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

#define CL_TARGET_OPENCL_VERSION 300

#if defined(COOT_USE_OPENCL)
  #if defined(__APPLE__)
    #include <OpenCL/opencl.h>
    #include <OpenCL/cl_platform.h>
  #else
    #include <CL/opencl.h>
    #include <CL/cl_platform.h>
  #endif

  #if defined(COOT_USE_CLBLAST)
    #include <clblast_c.h>
  #endif

  #if defined(COOT_USE_CLBLAS)
    #include <clBLAS.h>
  #endif
#endif
