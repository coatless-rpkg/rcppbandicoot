// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2026 Conrad Sanderson (https://conradsanderson.id.au)
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



#if defined(COOT_USE_OPENCL)
  
  #undef  CL_USE_DEPRECATED_OPENCL_1_2_APIS
  #define CL_USE_DEPRECATED_OPENCL_1_2_APIS
  
  #define CL_TARGET_OPENCL_VERSION COOT_TARGET_OPENCL_VERSION
  
  #if defined(__APPLE__)
    #include <OpenCL/opencl.h>
    #include <OpenCL/cl_platform.h>
  #else
    #include <CL/opencl.h>
    #include <CL/cl_platform.h>
  #endif
  
  #if defined(COOT_USE_CLBLAST)
    #if defined(__has_include)
      #if __has_include(<clblast_c.h>)
        #include <clblast_c.h>
      #else
        #undef COOT_USE_CLBLAST
        #pragma message ("WARNING: use of CLBlast disabled; clblast_c.h header not found")
      #endif
    #else
      #include <clblast_c.h>
    #endif
  #endif
  
  #if defined(COOT_USE_CLBLAS)
    #if defined(__has_include)
      #if __has_include(<clBLAS.h>)
        #include <clBLAS.h>
      #else
        #undef COOT_USE_CLBLAS
        #pragma message ("WARNING: use of clBLAS disabled; clBLAS.h header not found")
      #endif
    #else
      #include <clBLAS.h>
    #endif
  #endif
  
#endif
