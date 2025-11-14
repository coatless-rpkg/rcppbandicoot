// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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


// we need our own typedefs for types to use in template code;
// OpenCL attaches attributes to cl_int, cl_long, ...,
// which can cause lots of "attribute ignored" warnings
// when such types are used in template code


#if (UCHAR_MAX >= 0xff)
  typedef unsigned char      u8;
  typedef          char      s8;
#elif defined(UINT8_MAX)
  typedef          uint8_t   u8;
  typedef           int8_t   s8;
#else
  typedef          cl_uchar  u8;
  typedef          cl_char   s8;
#endif


#if (USHRT_MAX >= 0xffff)
  typedef unsigned short     u16;
  typedef          short     s16;
#elif defined(UINT16_MAX)
  typedef          uint16_t  u16;
  typedef           int16_t  s16;
#else
  typedef          cl_ushort u16;
  typedef          cl_short  s16;
#endif


#if (UINT_MAX >= 0xffffffff)
  typedef unsigned int      u32;
  typedef          int      s32;
#elif defined(UINT32_MAX)
  typedef          uint32_t u32;
  typedef           int32_t s32;
#else
  typedef          cl_uint  u32;
  typedef          cl_int   s32;
#endif


#if defined(UINT64_MAX)
  typedef          uint64_t  u64;
  typedef           int64_t  s64;
#elif (ULLONG_MAX == 0xffffffffffffffff)
  typedef unsigned long long u64;
  typedef          long long s64;
#elif (ULONG_MAX  == 0xffffffffffffffff)
  typedef unsigned long      u64;
  typedef          long      s64;
  #define COOT_U64_IS_LONG
#else
  typedef          cl_ulong  u64;
  typedef          cl_long   s64;
#endif


// need both signed and unsigned versions of size_t
typedef          std::size_t                         uword;
typedef typename std::make_signed<std::size_t>::type sword;


#if   defined(COOT_BLAS_LONG_LONG)
  typedef long long blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffffffffffULL
#elif defined(COOT_BLAS_LONG)
  typedef long      blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffffffffffUL
#else
  typedef int       blas_int;
  #define COOT_MAX_BLAS_INT 0x7fffffffU
#endif


typedef std::complex<float>  cx_float;
typedef std::complex<double> cx_double;

typedef void* void_ptr;



//


#if defined(COOT_USE_MKL_TYPES)
  // for compatibility with MKL
  typedef MKL_Complex8  blas_cxf;
  typedef MKL_Complex16 blas_cxd;
#else
  // standard BLAS and LAPACK prototypes use "void*" pointers for complex arrays
  typedef void blas_cxf;
  typedef void blas_cxd;
#endif


//


// NOTE: blas_len is the fortran type for "hidden" arguments that specify the length of character arguments;
// NOTE: it varies across compilers, compiler versions and systems (eg. 32 bit vs 64 bit);
// NOTE: the default setting of "size_t" is an educated guess.
// NOTE: ---
// NOTE: for gcc / gfortran:  https://gcc.gnu.org/onlinedocs/gfortran/Argument-passing-conventions.html
// NOTE: gcc 7 and earlier: int
// NOTE: gcc 8 and 9:       size_t
// NOTE: ---
// NOTE: for ifort (intel fortran compiler):~
// NOTE: "Intel Fortran Compiler User and Reference Guides", Document Number: 304970-006US, 2009, p. 301
// NOTE: http://www.complexfluids.ethz.ch/MK/ifort.pdf
// NOTE: the type is unsigned 4-byte integer on 32 bit systems
// NOTE: the type is unsigned 8-byte integer on 64 bit systems
// NOTE: ---
// NOTE: for NAG fortran: https://www.nag.co.uk/nagware/np/r62_doc/manual/compiler_11_1.html#AUTOTOC_11_1
// NOTE: Chrlen = usually int, or long long on 64-bit Windows
// NOTE: ---
// TODO: flang:  https://github.com/flang-compiler/flang/wiki
// TODO: other compilers: http://fortranwiki.org/fortran/show/Compilers

#if !defined(COOT_FORTRAN_CHARLEN_TYPE)
  #if defined(__GNUC__) && !defined(__clang__)
    #if (__GNUC__ <= 7)
      #define COOT_FORTRAN_CHARLEN_TYPE int
    #else
      #define COOT_FORTRAN_CHARLEN_TYPE size_t
    #endif
  #else
    // TODO: determine the type for other compilers
    #define COOT_FORTRAN_CHARLEN_TYPE size_t
  #endif
#endif

typedef COOT_FORTRAN_CHARLEN_TYPE blas_len;



//
// attempt to detect an FP16 type we can use, or if we find nothing, use
// uint16_t as a placeholder
//

#if defined(COOT_FP16_TYPE)
  typedef COOT_FP16_TYPE fp16;
#elif defined(COOT_HAVE_CXX23) && defined(__STDCPP_FLOAT16_T__) && (__STDCPP_FLOAT16_T__ == 1)
  // If we are using C++23, then we have an FP16 type via <stdfloat>, if hardware
  // support is available.
  typedef std::float16_t fp16;
#elif defined(COOT_USE_CUDA)
  typedef __half fp16;
#else
  // We don't have a half-precision type we can use for arithmetic on the host.
  // In this case, we will use our own "shim" type that actually holds a 32-bit
  // float on the host, but we will downcast it to a fp16 when dealing with the
  // GPU (and upcast when we bring back a value from the GPU).
  typedef fp16_shim fp16;
#endif
