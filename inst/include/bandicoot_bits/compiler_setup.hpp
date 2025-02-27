// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
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



#undef coot_hot
#undef coot_cold
#undef coot_aligned
#undef coot_align_mem
#undef coot_warn_unused
#undef coot_deprecated
#undef coot_malloc
#undef coot_inline
#undef coot_noinline
#undef coot_ignore

#define coot_hot
#define coot_cold
#define coot_aligned
#define coot_align_mem
#define coot_warn_unused
#define coot_deprecated
#define coot_malloc
#define coot_inline            inline
#define coot_noinline
#define coot_ignore(variable)  ((void)(variable))

#undef coot_fortran_noprefix
#undef coot_fortran_prefix

#undef coot_fortran2_noprefix
#undef coot_fortran2_prefix

#if defined(COOT_BLAS_UNDERSCORE)
  #define coot_fortran2_noprefix(function) function##_
  #define coot_fortran2_hidden_args_prefix(function)   wrapper_hidden_args_##function##_
  #define coot_fortran2_no_hidden_args_prefix(function)   wrapper_##function##_
#else
  #define coot_fortran2_noprefix(function) function
  #define coot_fortran2_hidden_args_prefix(function)   wrapper_hidden_args_##function##
  #define coot_fortran2_no_hidden_args_prefix(function)   wrapper_##function
#endif

#if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  #define coot_fortran2_prefix(function) coot_fortran2_hidden_args_prefix(function)
#else
  #define coot_fortran2_prefix(function) coot_fortran2_no_hidden_args_prefix(function)
#endif

#if defined(COOT_USE_WRAPPER)
  #define coot_fortran(function) coot_fortran2_prefix(function)
  #define coot_wrapper(function) wrapper_##function
#else
  #define coot_fortran(function) coot_fortran2_noprefix(function)
  #define coot_wrapper(function) function
#endif

#define coot_fortran_prefix(function)             coot_fortran2_prefix(function)
#define coot_fortran_noprefix(function)           coot_fortran2_noprefix(function)

#undef  COOT_INCFILE_WRAP
#define COOT_INCFILE_WRAP(x) <x>


#undef COOT_GOOD_COMPILER


#if ( defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L) )
  #undef  COOT_HAVE_POSIX_MEMALIGN
  #define COOT_HAVE_POSIX_MEMALIGN
#endif


#if defined(__APPLE__) || defined(__apple_build_version__)
  // #undef  COOT_HAVE_POSIX_MEMALIGN
  // NOTE: posix_memalign() is available since macOS 10.6 (late 2009 onwards)
#endif


#if defined(__APPLE__) || defined(__apple_build_version__)
  // NOTE: The Apple Accelerate framework uses a different convention for
  // linking FORTRAN functions, and so functions that return a float value
  // instead return a double.  We simply avoid using those functions (e.g.,
  // slange(), clange(), slanst(), slamc3()).
  #undef  COOT_FORTRAN_FLOAT_WORKAROUND
  #define COOT_FORTRAN_FLOAT_WORKAROUND
#endif


#if defined(__MINGW32__) || defined(__CYGWIN__) || defined(_MSC_VER)
  #undef COOT_HAVE_POSIX_MEMALIGN
#endif


#undef COOT_FNSIG

#if defined (__GNUG__)
  #define COOT_FNSIG  __PRETTY_FUNCTION__
#elif defined (_MSC_VER)
  #define COOT_FNSIG  __FUNCSIG__
#elif defined(__INTEL_COMPILER)
  #define COOT_FNSIG  __FUNCTION__
#else
  #define COOT_FNSIG  __func__
#endif


#if (defined(__GNUG__) || defined(__GNUC__)) && (defined(__clang__) || defined(__INTEL_COMPILER) || defined(__NVCC__) || defined(__CUDACC__) || defined(__PGI) || defined(__PATHSCALE__) || defined(__ARMCC_VERSION) || defined(__IBMCPP__))
  #undef  COOT_DETECTED_FAKE_GCC
  #define COOT_DETECTED_FAKE_GCC
#endif


#if defined(__GNUG__) && !defined(COOT_DETECTED_FAKE_GCC)

  #undef  COOT_GCC_VERSION
  #define COOT_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

  #if (COOT_GCC_VERSION < 40800)
    #error "*** newer compiler required; need gcc 4.8 or later ***"
  #endif

  #define COOT_GOOD_COMPILER

  #undef  coot_hot
  #undef  coot_cold
  #undef  coot_aligned
  #undef  coot_align_mem
  #undef  coot_warn_unused
  #undef  coot_deprecated
  #undef  coot_malloc
  #undef  coot_inline
  #undef  coot_noinline

  #define coot_hot         __attribute__((__hot__))
  #define coot_cold        __attribute__((__cold__))
  #define coot_aligned     __attribute__((__aligned__))
  #define coot_align_mem   __attribute__((__aligned__(16)))
  #define coot_warn_unused __attribute__((__warn_unused_result__))
  #define coot_deprecated  __attribute__((__deprecated__))
  #define coot_malloc      __attribute__((__malloc__))
  #define coot_inline      __attribute__((__always_inline__)) inline 
  #define coot_noinline    __attribute__((__noinline__))

#endif


#if defined(__clang__) && (defined(__INTEL_COMPILER) || defined(__NVCC__) || defined(__CUDACC__) || defined(__PGI) || defined(__PATHSCALE__) || defined(__ARMCC_VERSION) || defined(__IBMCPP__))
  #undef  COOT_DETECTED_FAKE_CLANG
  #define COOT_DETECTED_FAKE_CLANG
#endif


#if defined(__clang__) && !defined(COOT_DETECTED_FAKE_CLANG)

  #define COOT_GOOD_COMPILER

  #if !defined(__has_attribute)
    #define __has_attribute(x) 0
  #endif

  #if __has_attribute(__aligned__)
    #undef  coot_aligned
    #undef  coot_align_mem

    #define coot_aligned   __attribute__((__aligned__))
    #define coot_align_mem __attribute__((__aligned__(16)))

    #undef  COOT_HAVE_ALIGNED_ATTRIBUTE
    #define COOT_HAVE_ALIGNED_ATTRIBUTE
  #endif

  #if __has_attribute(__warn_unused_result__)
    #undef  coot_warn_unused
    #define coot_warn_unused __attribute__((__warn_unused_result__))
  #endif

  #if __has_attribute(__deprecated__)
    #undef  coot_deprecated
    #define coot_deprecated __attribute__((__deprecated__))
  #endif

  #if __has_attribute(__malloc__)
    #undef  coot_malloc
    #define coot_malloc __attribute__((__malloc__))
  #endif

  #if __has_attribute(__always_inline__)
    #undef  coot_inline
    #define coot_inline __attribute__((__always_inline__)) inline
  #endif

  #if __has_attribute(__noinline__)
    #undef  coot_noinline
    #define coot_noinline __attribute__((__noinline__))
  #endif

  #if __has_attribute(__hot__)
    #undef  coot_hot
    #define coot_hot __attribute__((__hot__))
  #endif

  #if __has_attribute(__cold__)
    #undef  coot_cold
    #define coot_cold __attribute__((__cold__))
  #elif __has_attribute(__minsize__)
    #undef  coot_cold
    #define coot_cold __attribute__((__minsize__))
  #endif

  // #pragma clang diagnostic push
  // #pragma clang diagnostic ignored "-Wignored-attributes"

#endif


#if defined(__INTEL_COMPILER)

  #if (__INTEL_COMPILER == 9999)
    #error "*** Need a newer compiler ***"
  #endif

  #if (__INTEL_COMPILER < 1500)
    #error "*** Need a newer compiler ***"
  #endif

#endif


#if defined(_MSC_VER)

  #if (_MSC_VER < 1900)
    #error "*** Need a newer compiler ***"
  #endif

  #undef  coot_deprecated
  #define coot_deprecated __declspec(deprecated)
  // #undef  coot_inline
  // #define coot_inline inline __forceinline

  #pragma warning(push)

  #pragma warning(disable: 4127)  // conditional expression is constant
  #pragma warning(disable: 4180)  // qualifier has no meaning
  #pragma warning(disable: 4244)  // possible loss of data when converting types
  #pragma warning(disable: 4510)  // default constructor could not be generated
  #pragma warning(disable: 4511)  // copy constructor can't be generated
  #pragma warning(disable: 4512)  // assignment operator can't be generated
  #pragma warning(disable: 4513)  // destructor can't be generated
  #pragma warning(disable: 4514)  // unreferenced inline function has been removed
  #pragma warning(disable: 4522)  // multiple assignment operators specified
  #pragma warning(disable: 4623)  // default constructor can't be generated
  #pragma warning(disable: 4624)  // destructor can't be generated
  #pragma warning(disable: 4625)  // copy constructor can't be generated
  #pragma warning(disable: 4626)  // assignment operator can't be generated
  #pragma warning(disable: 4702)  // unreachable code
  #pragma warning(disable: 4710)  // function not inlined
  #pragma warning(disable: 4711)  // call was inlined
  #pragma warning(disable: 4714)  // __forceinline can't be inlined
  #pragma warning(disable: 4800)  // value forced to bool

  // NOTE: also possible to disable 4146 (unary minus operator applied to unsigned type, result still unsigned)

  #if defined(COOT_HAVE_CXX17)
  #pragma warning(disable: 26812)  // unscoped enum
  #pragma warning(disable: 26819)  // unannotated fallthrough
  #endif

  // #if (_MANAGED == 1) || (_M_CEE == 1)
  //
  //   // don't do any alignment when compiling in "managed code" mode
  //
  //   #undef  coot_aligned
  //   #define coot_aligned
  //
  //   #undef  coot_align_mem
  //   #define coot_align_mem
  //
  // #elif (_MSC_VER >= 1700)
  //
  //   #undef  coot_align_mem
  //   #define coot_align_mem __declspec(align(16))
  //
  //   #define COOT_HAVE_ALIGNED_ATTRIBUTE
  //
  //   // disable warnings: "structure was padded due to __declspec(align(16))"
  //   #pragma warning(disable: 4324)
  //
  // #endif

#endif


#if defined(__SUNPRO_CC)

  // http://www.oracle.com/technetwork/server-storage/solarisstudio/training/index-jsp-141991.html
  // http://www.oracle.com/technetwork/server-storage/solarisstudio/documentation/cplusplus-faq-355066.html

  #if (__SUNPRO_CC < 0x5140)
    #error "*** Need a newer compiler ***"
  #endif

#endif


#if defined(COOT_HAVE_CXX14)
  #undef  coot_deprecated
  #define coot_deprecated [[deprecated]]

  #undef  coot_frown
  #define coot_frown(msg) [[deprecated(msg)]]
#endif


#if defined(COOT_HAVE_CXX17)
  #undef  coot_warn_unused
  #define coot_warn_unused  [[nodiscard]]
#endif


#if !defined(COOT_DONT_USE_OPENMP)
  #if (defined(_OPENMP) && (_OPENMP >= 201107))
    #undef  COOT_USE_OPENMP
    #define COOT_USE_OPENMP
  #endif
#endif


#if ( defined(COOT_USE_OPENMP) && (!defined(_OPENMP) || (defined(_OPENMP) && (_OPENMP < 201107))) )
  // OpenMP 3.0 required for parallelisation of loops with unsigned integers
  // OpenMP 3.1 required for atomic read and atomic write
  #undef  COOT_USE_OPENMP
  #undef  COOT_PRINT_OPENMP_WARNING
  #define COOT_PRINT_OPENMP_WARNING
#endif


#if defined(COOT_PRINT_OPENMP_WARNING) && !defined(COOT_DONT_PRINT_OPENMP_WARNING)
  #pragma message ("WARNING: use of OpenMP disabled; compiler support for OpenMP 3.1+ not detected")
  
  #if (defined(_OPENMP) && (_OPENMP < 201107))
    #pragma message ("NOTE: your compiler has an outdated version of OpenMP")
    #pragma message ("NOTE: consider upgrading to a better compiler")
  #endif
#endif


#if defined(COOT_USE_OPENMP)
  #if (defined(COOT_GCC_VERSION) && (COOT_GCC_VERSION < 50400))
    // due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57580
    #undef COOT_USE_OPENMP
    #if !defined(COOT_DONT_PRINT_OPENMP_WARNING)
      #pragma message ("WARNING: use of OpenMP disabled due to compiler bug in gcc <= 5.3")
    #endif
  #endif
#endif


#if ( (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)) && (!defined(__MINGW32__) && !defined(__MINGW64__)) )
  #undef  COOT_PRINT_EXCEPTIONS_INTERNAL
  #define COOT_PRINT_EXCEPTIONS_INTERNAL
#endif


// cleanup

#undef COOT_DETECTED_FAKE_GCC
#undef COOT_DETECTED_FAKE_CLANG
#undef COOT_GCC_VERSION
#undef COOT_PRINT_OPENMP_WARNING



// undefine conflicting macros

#if defined(log2)
  #undef log2
  #pragma message ("WARNING: undefined conflicting 'log2' macro")
#endif

#if defined(check)
  #undef check
  #pragma message ("WARNING: undefined conflicting 'check' macro")
#endif

#if defined(min) || defined(max)
  #undef min
  #undef max
  #pragma message ("WARNING: undefined conflicting 'min' and/or 'max' macros;")
  #pragma message ("WARNING: suggest to define NOMINMAX before including any windows header")
#endif

// https://sourceware.org/bugzilla/show_bug.cgi?id=19239
#undef minor
#undef major
