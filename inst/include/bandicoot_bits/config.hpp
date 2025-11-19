// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017-2023 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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

#if !defined(COOT_WARN_LEVEL)
  #define COOT_WARN_LEVEL 2
#endif
//// The level of warning messages printed to COOT_CERR_STREAM.
//// Must be an integer >= 0. The default value is 2.
//// 0 = no warnings; generally not recommended
//// 1 = only critical warnings about arguments and/or data which are likely to lead to incorrect results
//// 2 = as per level 1, and warnings about poorly conditioned systems (low rcond) detected by solve() etc
//// 3 = as per level 2, and warnings about failed decompositions, failed saving/loading, etc

// #define COOT_USE_WRAPPER
//// Comment out the above line if you prefer to directly link with CUDA, OpenCL, OpenBLAS, etc
//// instead of the Bandicoot runtime library.

#if !defined(COOT_USE_OPENCL)
#define COOT_USE_OPENCL
//// Uncomment the above line if you have OpenCL available on your system.
//// Bandicoot requires OpenCL to be available.
//// For nontrivial operations it is also required to have either clBLAS or CLBlast available also.
#endif

#if !defined(COOT_USE_CLBLAST)
#define COOT_USE_CLBLAST
//// Uncomment the above line if you have CLBlast available on your system.
#endif

#if !defined(COOT_USE_CLBLAS)
#define COOT_USE_CLBLAS
//// Uncomment the above line if you have clBLAS available on your system.
//// It is generally preferred to use CLBlast.
#endif

#if !defined(COOT_USE_CUDA)
#define COOT_USE_CUDA
//// Uncomment the above line if you have CUDA available on your system.
//// Bandicoot requires CUDA, CUDART, cuBLAS, cuRAND, cuSolver, and NVRTC.
#define COOT_CUDA_INCLUDE_PATH /usr/include/
//// Set the above line to the include path used to include parts of the CUDA toolkit.
//// These will be used by NVRTC to compile kernels on-the-fly.
#endif

#if !defined(COOT_DEFAULT_BACKEND)
#define COOT_DEFAULT_BACKEND CL_BACKEND
//// This defines the backend that Bandicoot will use by default.
//// It takes values either CL_BACKEND or CUDA_BACKEND;
//// if set to CL_BACKEND, then COOT_USE_OPENCL must be defined;
//// if set to CUDA_BACKEND, then COOT_USE_CUDA must be defined.
#endif

#if !defined(COOT_USE_LAPACK)
#define COOT_USE_LAPACK
//// Comment out the above line if you don't have LAPACK or a high-speed replacement for LAPACK,
//// such as OpenBLAS, Intel MKL, or the Accelerate framework.
#endif

#if !defined(COOT_USE_BLAS)
#define COOT_USE_BLAS
//// Comment out the above line if you don't have BLAS or a high-speed replacement for BLAS,
//// such as OpenBLAS, Intel MKL, or the Accelerate framework.
#endif

// #define COOT_BLAS_CAPITALS
//// Uncomment the above line if your BLAS and LAPACK libraries have capitalised function names

#define COOT_BLAS_UNDERSCORE
//// Uncomment the above line if your BLAS and LAPACK libraries have function names with a trailing underscore.
//// Conversely, comment it out if the function names don't have a trailing underscore.

// #define COOT_BLAS_LONG
//// Uncomment the above line if your BLAS and LAPACK libraries use "long" instead of "int"

// #define COOT_BLAS_LONG_LONG
//// Uncomment the above line if your BLAS and LAPACK libraries use "long long" instead of "int"

// #define COOT_BLAS_NOEXCEPT
//// Uncomment the above line if you require BLAS functions to have the 'noexcept' specification

// #define COOT_LAPACK_NOEXCEPT
//// Uncomment the above line if you require LAPACK functions to have the 'noexcept' specification

#if !defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
#define COOT_USE_FORTRAN_HIDDEN_ARGS
//// Comment out the above line to call BLAS and LAPACK functions without using so-called "hidden" arguments.
//// FORTRAN functions (compiled without a BIND(C) declaration) that have char arguments
//// (like many BLAS and LAPACK functions) also have associated "hidden" arguments.
//// For each char argument, the corresponding "hidden" argument specifies the number of characters.
//// These "hidden" arguments are typically tacked onto the end of function definitions.
#endif

// #define COOT_USE_MKL_TYPES
//// Uncomment the above line if you want to use Intel MKL types for complex numbers.
//// You will need to include appropriate MKL headers before the Bandicoot header.
//// You may also need to enable or disable the following options:
//// COOT_BLAS_LONG, COOT_BLAS_LONG_LONG, COOT_USE_FORTRAN_HIDDEN_ARGS

#if !defined(COOT_USE_OPENMP)
// #define COOT_USE_OPENMP
//// Uncomment the above line to forcefully enable use of OpenMP for parallelisation.
//// Note that COOT_USE_OPENMP is automatically enabled when a compiler supporting OpenMP 3.1 is detected.
#endif

// #define COOT_NO_DEBUG
//// Uncomment the above line to disable all run-time checks. NOT RECOMMENDED.
//// It is strongly recommended that run-time checks are enabled during development,
//// as this greatly aids in finding mistakes in your code.

// #define COOT_EXTRA_DEBUG
//// Uncomment the above line if you want to see the function traces of how Bandicoot evaluates expressions.
//// This is mainly useful for debugging of the library.

#if defined(COOT_EXTRA_DEBUG)
  #undef  COOT_NO_DEBUG
  #undef  COOT_WARN_LEVEL
  #define COOT_WARN_LEVEL 3
#endif

#if !defined(COOT_COUT_STREAM)
  #define COOT_COUT_STREAM std::cout
#endif

#if !defined(COOT_CERR_STREAM)
  #define COOT_CERR_STREAM std::cerr
#endif

#if !defined(COOT_PRINT_EXCEPTIONS)
  // #define COOT_PRINT_EXCEPTIONS
  #if defined(COOT_PRINT_EXCEPTIONS_INTERNAL)
    #undef  COOT_PRINT_EXCEPTIONS
    #define COOT_PRINT_EXCEPTIONS
  #endif
#endif

#if defined(COOT_DONT_USE_LAPACK)
  #undef COOT_USE_LAPACK
#endif

#if defined(COOT_DONT_USE_BLAS)
  #undef COOT_USE_BLAS
#endif

#if defined(COOT_DONT_USE_OPENCL)
  #undef COOT_USE_OPENCL
#endif

#if defined(COOT_DONT_USE_CLBLAST)
  #undef COOT_USE_CLBLAST
#endif

#if defined(COOT_DONT_USE_CLBLAS)
  #undef COOT_USE_CLBLAS
#endif

#if defined(COOT_DONT_USE_CUDA)
  #undef COOT_USE_CUDA
#endif

#if defined(COOT_DONT_USE_WRAPPER)
  #undef COOT_USE_WRAPPER
#endif

#if defined(COOT_DONT_USE_OPENMP)
  #undef COOT_USE_OPENMP
#endif

#if defined(COOT_DONT_PRINT_EXCEPTIONS)
  #undef COOT_PRINT_EXCEPTIONS
#endif

#if !defined(COOT_DEFAULT_BACKEND)
  #if defined(COOT_USE_OPENCL)
    #define COOT_DEFAULT_BACKEND CL_BACKEND
  #elif defined(COOT_USE_CUDA)
    #define COOT_DEFAULT_BACKEND CUDA_BACKEND
  #else
    #error "One of COOT_USE_OPENCL or COOT_USE_CUDA must be defined!"
  #endif
#else
  // TODO: ensure that the backend is valid
#endif


// Uncomment and modify the lines below to specify a custom directory to store Bandicoot kernels to.
// Alternately, define COOT_KERNEL_CACHE_DIR in your program.
// Note that COOT_KERNEL_CACHE_DIR must have a / as its final character (or \ on Windows).
//
// #if defined(COOT_KERNEL_CACHE_DIR)
//   #undef COOT_KERNEL_CACHE_DIR
//   #define COOT_KERNEL_CACHE_DIR /custom/cache/location/
// #endif

// Set default location of system-wide kernel cache on Linux.
#if !defined(COOT_SYSTEM_KERNEL_CACHE_DIR)
  #if __linux__
    #define COOT_SYSTEM_KERNEL_CACHE_DIR "/var/cache/bandicoot/"
  #endif
#endif

// Uncomment and modify the lines below to specify a custom directory where Bandicoot kernel sources are stored.
// Alternately, define COOT_KERNEL_SOURCE_DIR in your program.
// Note that COOT_KERNEL_SOURCE_DIR must have a / as its final character (or \ on Windows).
//
// Kernels are expected to be in the following structure:
//   COOT_KERNEL_SOURCE_DIR/
//     cuda/
//       defs/ (these directories contain the individual .cu files)
//       oneway/
//       .../
//     opencl/
//       defs/ (these directories contain the individual .cl files)
//       oneway/
//       .../
//
// #if defined(COOT_KERNEL_SOURCE_DIR)
//   #undef COOT_KERNEL_SOURCE_DIR
//   #define COOT_KERNEL_SOURCE_DIR /custom/kernel/location/
// #endif

// if Bandicoot was installed on this system via CMake and COOT_USE_WRAPPER is not defined,
// COOT_AUX_LIBS lists the libraries required by Bandicoot on this system, and
// COOT_AUX_INCDIRS lists the include directories required by Bandicoot on this system.
// Do not use these unless you know what you are doing.
#define COOT_AUX_LIBS
#define COOT_AUX_INCDIRS
