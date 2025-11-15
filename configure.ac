##
## RcppBandicoot configure.ac
##
## Copyright (C) 2023-2025 James Balamuta
##
## Licensed under GPL-2 or later
##

AC_INIT([RcppBandicoot], [0.1.11.1.9000])

: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME])
fi
CC=$(${R_HOME}/bin/R CMD config CC)
CXX=$(${R_HOME}/bin/R CMD config CXX)
CFLAGS=$(${R_HOME}/bin/R CMD config CFLAGS)
CPPFLAGS=$(${R_HOME}/bin/R CMD config CPPFLAGS)
CXXFLAGS=$(${R_HOME}/bin/R CMD config CXXFLAGS)
LDFLAGS=$(${R_HOME}/bin/R CMD config LDFLAGS)

AC_LANG(C++)
AC_REQUIRE_CPP
AC_PROG_CC
AC_PROG_CXX

## =============================================================================
## Platform Detection (macOS vs Linux/Unix)
## =============================================================================

AC_MSG_CHECKING([for macOS])
RSysinfoName=$("${R_HOME}/bin/Rscript" --vanilla -e 'cat(Sys.info()[["sysname"]])')
if test x"${RSysinfoName}" = x"Darwin"; then
    AC_MSG_RESULT([found])
    is_macos="yes"

    AC_MSG_CHECKING([for macOS Apple compiler])
    apple_compiler=$($CXX --version 2>&1 | grep -i -c -e 'apple llvm')
    if test x"${apple_compiler}" = x"1"; then
        AC_MSG_RESULT([found])
        is_apple_compiler="yes"
    else
        AC_MSG_RESULT([not found])
        is_apple_compiler="no"
    fi
else
    AC_MSG_RESULT([not found])
    is_macos="no"
    is_apple_compiler="no"
fi

## =============================================================================
## C++14 Standard Check (Required by Bandicoot)
## =============================================================================
## Note: C++14 is specified in SystemRequirements in DESCRIPTION
## R's build system will automatically use the appropriate C++ standard
## We use C++14 here for configure tests to ensure dependency compatibility

AC_MSG_CHECKING([whether C++14 is supported])
CXX14=$(${R_HOME}/bin/R CMD config CXX14)
CXX14FLAGS=$(${R_HOME}/bin/R CMD config CXX14FLAGS)
CXX14STD=$(${R_HOME}/bin/R CMD config CXX14STD)

if test "x${CXX14}" = x; then
  AC_MSG_ERROR([C++14 support is required but not available. Please use a compiler that supports C++14.])
fi
AC_MSG_RESULT([yes])

## Use C++14 for configure tests (to ensure dependency compatibility)
CXX="${CXX14} ${CXX14STD}"
CXXFLAGS="${CXX14FLAGS} ${CXXFLAGS}"

## Default flags
BANDICOOT_CXXFLAGS=""
BANDICOOT_LIBS=""
HAVE_OPENCL=0
HAVE_CUDA=0
HAVE_CLBLAST=0
HAVE_CLBLAS=0
DEFAULT_BACKEND="NONE"

## =============================================================================
## OpenCL Detection
## =============================================================================

AC_MSG_CHECKING([for OpenCL])

## Check for OpenCL headers (platform-specific)
opencl_header_found=no

## Platform-specific OpenCL detection
if test x"${is_macos}" = x"yes"; then
    ## macOS: OpenCL is a system framework
    AC_MSG_RESULT([checking macOS OpenCL framework])

    saved_CPPFLAGS="${CPPFLAGS}"
    saved_LIBS="${LIBS}"

    ## Find the SDK path and add framework search path
    SDKPATH=$(xcrun --show-sdk-path 2>/dev/null)
    if test -n "${SDKPATH}"; then
      FRAMEWORK_SEARCH_PATH="${SDKPATH}/System/Library/Frameworks"
      if test -d "${FRAMEWORK_SEARCH_PATH}"; then
        CPPFLAGS="${CPPFLAGS} -F${FRAMEWORK_SEARCH_PATH}"
      fi
    fi

    ## Try to compile a test program using OpenCL framework
    LIBS="-framework OpenCL ${saved_LIBS}"

    AC_MSG_NOTICE([Detected SDK path: ${SDKPATH}])
    AC_MSG_NOTICE([Framework search path: ${FRAMEWORK_SEARCH_PATH}])

    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
      #include <OpenCL/opencl.h>
    ]], [[
      cl_platform_id platform;
      clGetPlatformIDs(1, &platform, NULL);
    ]])],
    [opencl_header_found=yes
     HAVE_OPENCL=1
     BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_USE_OPENCL"
     if test -n "${SDKPATH}" && test -d "${FRAMEWORK_SEARCH_PATH}"; then
       BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -F${FRAMEWORK_SEARCH_PATH}"
     fi
     BANDICOOT_LIBS="${BANDICOOT_LIBS} -framework OpenCL"
     AC_MSG_RESULT([yes (framework)])],
    [opencl_header_found=no
     HAVE_OPENCL=0
     AC_MSG_RESULT([no - OpenCL framework not found or incomplete])])

    LIBS="${saved_LIBS}"
    CPPFLAGS="${saved_CPPFLAGS}"

    ## If OpenCL not found on macOS, provide helpful guidance
    if test "x${HAVE_OPENCL}" = x0; then
      AC_MSG_WARN([macOS OpenCL framework not found or not functional])
      AC_MSG_WARN([You may need to install Xcode Command Line Tools: xcode-select --install])
      AC_MSG_WARN([Or use CUDA if you have an NVIDIA GPU])
    fi
else
    ## Linux/Unix: Check for CL/opencl.h header
    AC_CHECK_HEADER([CL/opencl.h],
                    [opencl_header_found=yes
                     OPENCL_INCLUDE="<CL/opencl.h>"
                     BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_USE_OPENCL"],
                    [opencl_header_found=no])

    ## If headers found, check for OpenCL library
    if test "x${opencl_header_found}" = xyes; then
      saved_LIBS="${LIBS}"
      LIBS="-lOpenCL ${saved_LIBS}"

      AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include ${OPENCL_INCLUDE}]],
                                      [[clGetPlatformIDs(0,0,0);]])],
                     [HAVE_OPENCL=1
                      BANDICOOT_LIBS="${BANDICOOT_LIBS} -lOpenCL"
                      AC_MSG_RESULT([yes (-lOpenCL)])],
                     [HAVE_OPENCL=0
                      AC_MSG_RESULT([no])
                      BANDICOOT_CXXFLAGS=$(echo ${BANDICOOT_CXXFLAGS} | sed 's/-DCOOT_USE_OPENCL//')])

      LIBS="${saved_LIBS}"
    else
      AC_MSG_RESULT([no])
    fi
fi

## =============================================================================
## CLBlast Detection (preferred for OpenCL)
## =============================================================================

if test "x${HAVE_OPENCL}" = x1; then
  AC_MSG_CHECKING([for CLBlast])

  saved_LIBS="${LIBS}"
  saved_LDFLAGS="${LDFLAGS}"
  saved_CPPFLAGS="${CPPFLAGS}"

  ## On macOS, check for Homebrew installation
  if test x"${is_macos}" = x"yes"; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null)
    if test -n "${BREW_PREFIX}" && test -d "${BREW_PREFIX}/opt/clblast"; then
      CLBLAST_PREFIX="${BREW_PREFIX}/opt/clblast"
      AC_MSG_NOTICE([Found Homebrew CLBlast at: ${CLBLAST_PREFIX}])
      LDFLAGS="${LDFLAGS} -L${CLBLAST_PREFIX}/lib"
      CPPFLAGS="${CPPFLAGS} -I${CLBLAST_PREFIX}/include"
    fi
  fi

  LIBS="-lclblast ${BANDICOOT_LIBS} ${saved_LIBS}"

  AC_LINK_IFELSE([AC_LANG_PROGRAM([[extern "C" void CLBlastSgemm();]],
                                  [[CLBlastSgemm();]])],
                 [HAVE_CLBLAST=1
                  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_USE_CLBLAST"
                  if test -n "${CLBLAST_PREFIX}"; then
                    BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -I${CLBLAST_PREFIX}/include"
                    BANDICOOT_LIBS="-L${CLBLAST_PREFIX}/lib -lclblast ${BANDICOOT_LIBS}"
                  else
                    BANDICOOT_LIBS="-lclblast ${BANDICOOT_LIBS}"
                  fi
                  AC_MSG_RESULT([yes])],
                 [HAVE_CLBLAST=0
                  ## Explicitly disable CLBlast since bandicoot's config.hpp enables it by default
                  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_DONT_USE_CLBLAST"
                  AC_MSG_RESULT([no])])

  LIBS="${saved_LIBS}"
  LDFLAGS="${saved_LDFLAGS}"
  CPPFLAGS="${saved_CPPFLAGS}"
fi

## =============================================================================
## clBLAS Detection (used alongside CLBlast by bandicoot)
## =============================================================================
## Note: Bandicoot's config.hpp enables both CLBLAST and CLBLAS by default,
## so we need to provide headers for both even if we prefer CLBlast

if test "x${HAVE_OPENCL}" = x1; then
  AC_MSG_CHECKING([for clBLAS])

  saved_LIBS="${LIBS}"
  saved_LDFLAGS="${LDFLAGS}"
  saved_CPPFLAGS="${CPPFLAGS}"

  ## On macOS, check for Homebrew installation
  if test x"${is_macos}" = x"yes"; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null)
    if test -n "${BREW_PREFIX}" && test -d "${BREW_PREFIX}/opt/clblas"; then
      CLBLAS_PREFIX="${BREW_PREFIX}/opt/clblas"
      AC_MSG_NOTICE([Found Homebrew clBLAS at: ${CLBLAS_PREFIX}])
      LDFLAGS="${LDFLAGS} -L${CLBLAS_PREFIX}/lib"
      CPPFLAGS="${CPPFLAGS} -I${CLBLAS_PREFIX}/include"
    fi
  fi

  LIBS="-lclBLAS ${BANDICOOT_LIBS} ${saved_LIBS}"

  AC_LINK_IFELSE([AC_LANG_PROGRAM([[extern "C" void clblasSetup();]],
                                  [[clblasSetup();]])],
                 [HAVE_CLBLAS=1
                  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_USE_CLBLAS"
                  if test -n "${CLBLAS_PREFIX}"; then
                    BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -I${CLBLAS_PREFIX}/include"
                    BANDICOOT_LIBS="-L${CLBLAS_PREFIX}/lib -lclBLAS ${BANDICOOT_LIBS}"
                  else
                    BANDICOOT_LIBS="-lclBLAS ${BANDICOOT_LIBS}"
                  fi
                  AC_MSG_RESULT([yes])],
                 [HAVE_CLBLAS=0
                  ## Explicitly disable clBLAS since bandicoot's config.hpp enables it by default
                  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_DONT_USE_CLBLAS"
                  AC_MSG_RESULT([no])])

  LIBS="${saved_LIBS}"
  LDFLAGS="${saved_LDFLAGS}"
  CPPFLAGS="${saved_CPPFLAGS}"
fi

## Check if OpenCL has at least one BLAS implementation
if test "x${HAVE_OPENCL}" = x1 && test "x${HAVE_CLBLAST}" = x0 && test "x${HAVE_CLBLAS}" = x0; then
  AC_MSG_WARN([OpenCL detected but no OpenCL BLAS library (CLBlast or clBLAS) found.])
  AC_MSG_WARN([Performance will be severely limited. Please install CLBlast.])
  AC_MSG_WARN([Ubuntu/Debian: sudo apt-get install libclblast-dev])
  AC_MSG_WARN([Fedora/RHEL: sudo dnf install clblast-devel])
  AC_MSG_WARN([macOS: brew install clblast])
fi

## =============================================================================
## CUDA Detection
## =============================================================================

AC_MSG_CHECKING([for CUDA])

## Look for nvcc
AC_PATH_PROG([NVCC], [nvcc], [no])

if test "x${NVCC}" != xno; then
  ## Found nvcc, now check for CUDA headers and libraries
  CUDA_HOME=$(dirname $(dirname ${NVCC}))

  ## Check for cuda_runtime.h
  saved_CPPFLAGS="${CPPFLAGS}"
  CPPFLAGS="${CPPFLAGS} -I${CUDA_HOME}/include"

  AC_CHECK_HEADER([cuda_runtime.h],
                  [cuda_header_found=yes],
                  [cuda_header_found=no])

  if test "x${cuda_header_found}" = xyes; then
    ## Check for CUDA libraries
    saved_LIBS="${LIBS}"
    saved_LDFLAGS="${LDFLAGS}"
    LDFLAGS="${LDFLAGS} -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib"

    AC_CHECK_LIB([cudart], [cudaMalloc],
                 [AC_CHECK_LIB([cublas], [cublasCreate],
                  [AC_CHECK_LIB([cusolver], [cusolverDnCreate],
                   [AC_CHECK_LIB([curand], [curandCreateGenerator],
                    [AC_CHECK_LIB([nvrtc], [nvrtcCreateProgram],
                     [HAVE_CUDA=1
                      BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_USE_CUDA -DCOOT_CUDA_INCLUDE_PATH=${CUDA_HOME}/include/"
                      BANDICOOT_LIBS="${BANDICOOT_LIBS} -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib -lcudart -lcublas -lcusolver -lcurand -lnvrtc"
                      AC_MSG_RESULT([yes])],
                     [HAVE_CUDA=0])],
                    [HAVE_CUDA=0])],
                   [HAVE_CUDA=0])],
                  [HAVE_CUDA=0])],
                 [HAVE_CUDA=0])

    LIBS="${saved_LIBS}"
    LDFLAGS="${saved_LDFLAGS}"
  fi

  CPPFLAGS="${saved_CPPFLAGS}"
fi

if test "x${HAVE_CUDA}" = x0; then
  AC_MSG_RESULT([no])
  ## Explicitly disable CUDA since bandicoot's config.hpp enables it by default
  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_DONT_USE_CUDA"
fi

## =============================================================================
## Determine Default Backend
## =============================================================================

if test "x${HAVE_OPENCL}" = x1; then
  DEFAULT_BACKEND="CL_BACKEND"
  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_DEFAULT_BACKEND=CL_BACKEND"
elif test "x${HAVE_CUDA}" = x1; then
  DEFAULT_BACKEND="CUDA_BACKEND"
  BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} -DCOOT_DEFAULT_BACKEND=CUDA_BACKEND"
else
  AC_MSG_ERROR([
================================================================================
ERROR: No GPU backend detected
================================================================================

RcppBandicoot requires at least one GPU backend (OpenCL or CUDA).

For OpenCL support, install:
  Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev libclblast-dev
  Fedora/RHEL:   sudo dnf install opencl-headers ocl-icd-devel clblast-devel
  macOS:         OpenCL is built-in; install CLBlast with: brew install clblast

For CUDA support, install:
  CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
  Minimum version: CUDA 9.8

After installation, ensure GPU drivers are properly installed and configured.
================================================================================
])
fi

AC_MSG_NOTICE([])
AC_MSG_NOTICE([GPU Backend Configuration Summary:])
AC_MSG_NOTICE([  Platform:     ${RSysinfoName}])
AC_MSG_NOTICE([  C++ Standard: C++14])
AC_MSG_NOTICE([  OpenCL:       ${HAVE_OPENCL}])
if test "x${HAVE_OPENCL}" = x1; then
  AC_MSG_NOTICE([  CLBlast:      ${HAVE_CLBLAST}])
  AC_MSG_NOTICE([  clBLAS:       ${HAVE_CLBLAS}])
fi
AC_MSG_NOTICE([  CUDA:         ${HAVE_CUDA}])
AC_MSG_NOTICE([  Default:      ${DEFAULT_BACKEND}])
AC_MSG_NOTICE([])

## =============================================================================
## OpenMP Detection (from R's configuration)
## =============================================================================

AC_MSG_CHECKING([for OpenMP])

## Check if Apple compiler, which typically doesn't support OpenMP
if test x"${is_apple_compiler}" = x"yes"; then
  AC_MSG_RESULT([unavailable (Apple compiler)])
  AC_MSG_WARN([OpenMP unavailable and turned off.])
  OPENMP_CXXFLAGS=""
else
  ## Check if R has -fopenmp in ldflags
  allldflags=$(${R_HOME}/bin/R CMD config --ldflags)
  hasOpenMP=$(echo ${allldflags} | grep -- -fopenmp)
  if test x"${hasOpenMP}" = x""; then
    AC_MSG_RESULT([missing])
    OPENMP_CXXFLAGS=""
  else
    AC_MSG_RESULT([found and suitable])
    OPENMP_CXXFLAGS='$(SHLIB_OPENMP_CXXFLAGS)'
    BANDICOOT_CXXFLAGS="${BANDICOOT_CXXFLAGS} ${OPENMP_CXXFLAGS}"
  fi
fi

## =============================================================================
## LAPACK/BLAS (from R's configuration)
## =============================================================================

LAPACK_LIBS=$(${R_HOME}/bin/R CMD config LAPACK_LIBS)
BLAS_LIBS=$(${R_HOME}/bin/R CMD config BLAS_LIBS)
FLIBS=$(${R_HOME}/bin/R CMD config FLIBS)

BANDICOOT_LIBS="${BANDICOOT_LIBS} ${LAPACK_LIBS} ${BLAS_LIBS} ${FLIBS}"

## =============================================================================
## Generate output
## =============================================================================

AC_SUBST([BANDICOOT_CXXFLAGS])
AC_SUBST([BANDICOOT_LIBS])
AC_SUBST([OPENMP_CXXFLAGS])

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
