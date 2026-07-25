# RcppBandicoot

[![R-CMD-check](https://github.com/coatless-rpkg/rcppbandicoot/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/coatless-rpkg/rcppbandicoot/actions/workflows/R-CMD-check.yaml)

## Overview

[Bandicoot](https://coot.sourceforge.io/) is a C++ header-only GPU accelerated
linear algebra library written by developers behind [Armadillo](https://arma.sourceforge.net/) that provides high-level syntax for performing computations on
graphics processing units.

The RcppBandicoot package includes the header files from the Bandicoot library
and integrates a seamless experience with R by using the 
[Rcpp](https://cran.r-project.org/package=Rcpp) package. 
Therefore, users do not need to install Bandicoot to use RcppBandicoot. 

## Installation

You can install RcppBandicoot from GitHub:

```r
# install.packages("remotes")
remotes::install_github("coatless-rpkg/rcppbandicoot")
```

Installing from source detects and records whatever OpenCL SDK and BLAS library
are present at that moment; see [Requirements](#requirements) below. Note that
having the package installed is not the same as having a device to run on --
see [Getting an OpenCL device](#getting-an-opencl-device).

## Usage from R

```r
library(RcppBandicoot)

bandicoot_config()   # backends and BLAS libraries this build was compiled with
gpu_available()      # is there a device that can actually compile and run a kernel?

A <- matrix(rnorm(100), 10, 10)
gpu_matrix_multiply(A, A)
```

Call **at most one** of `gpu_available()`, `gpu_initialize()` and
`gpu_device_info()`, and call it **at most once per session**. Each is a thin
wrapper around Bandicoot's `coot_init()`, which unconditionally tears down the
live OpenCL context and builds a fresh one without evicting the kernels already
compiled against the old one. The second such call in a session leaves dangling
kernel handles, after which every device query reports no device and every
operation fails with `cl_invalid_context`. Ordinary operations
(`gpu_matrix_multiply()`, `gpu_eye()`, ...) reach the runtime through a guarded
path and are unaffected.

## Usage from your own package

Add RcppBandicoot to `DESCRIPTION`:

```
LinkingTo: Rcpp, RcppBandicoot
Imports: Rcpp
```

and add a `src/Makevars` (copy it to `src/Makevars.win` as well):

```make
PKG_CPPFLAGS = $(shell "$(R_HOME)/bin/Rscript" -e "RcppBandicoot::CxxFlags()")
PKG_LIBS     = $(shell "$(R_HOME)/bin/Rscript" -e "RcppBandicoot::LdFlags()")
```

**The `Makevars` is not optional.** `LinkingTo` contributes an `-I` include path
and nothing else, so building against the headers alone fails three times in a
row:

1. at compile time, with `One of COOT_USE_OPENCL, COOT_USE_CUDA, or COOT_USE_VULKAN must be defined!`
   -- a `#error` in `bandicoot_bits/config.hpp`, because the backend macro lives
   in the flags, not in the headers;
2. at link time, with undefined references to `clGetPlatformIDs` and friends --
   `-lOpenCL` and `-lclblast` live in the flags too;
3. at run time, with `Cannot open required kernel source.` -- Bandicoot reads
   its `.cl` kernels from disk, from the directory named by
   `COOT_KERNEL_SOURCE_DIR`, and without that macro it looks for a path this
   package does not ship.

Use `CxxFlags()` and `LdFlags()`, not `RcppBandicootCxxFlags()` and
`RcppBandicootLdFlags()`. The two pairs return the same strings, but the
`RcppBandicoot`-prefixed ones only *return* theirs, so `Rscript -e` auto-prints
them as `[1] "-I'/path' -DCOOT_KERNEL_SOURCE_DIR='\"/path/\"' ..."` and `make`
would paste the index prefix, the surrounding quotes and the backslash escapes
straight onto the compile line. `CxxFlags()` and `LdFlags()` `cat()` their value
and return it invisibly, so standard output is exactly the flags.

Then include the main header:

```cpp
#include <RcppBandicoot.h>

// [[Rcpp::depends(RcppBandicoot)]]

// [[Rcpp::export]]
coot::mat gpu_multiply(const coot::mat& A, const coot::mat& B) {
    return A * B;
}
```

## Usage from sourceCpp()

`Rcpp::sourceCpp()` needs no `Makevars`. The `// [[Rcpp::depends(RcppBandicoot)]]`
attribute makes Rcpp call this package's `inlineCxxPlugin()`, which supplies the
same compiler and linker flags automatically.

## Conversions

The RcppBandicoot integration provides automatic conversion between:
- R matrices ↔ `coot::Mat<T>`
- R vectors ↔ `coot::Col<T>` and `coot::Row<T>`
- R 3D arrays ↔ `coot::Cube<T>`

Operations on `coot::` types run on whichever device Bandicoot selected, which
may be a discrete GPU or a CPU OpenCL runtime. Every conversion copies data
between R's memory and the device, so a single small operation is usually slower
than its base R equivalent; the benefit comes from keeping large data resident on
the device across a sequence of operations.

## Getting an OpenCL device

This is the part that most often goes wrong. An OpenCL program needs two
separate things: the **ICD loader** (`libOpenCL.so`, `OpenCL.dll`, or Apple's
OpenCL framework), which is what RcppBandicoot links against, and a **vendor
runtime** that registers an actual device with that loader. Installing the
package gives you the first. It cannot give you the second.

`clinfo` is the quickest way to see what the loader can find; if it reports zero
platforms or zero devices, no R-side change will help.

### No device is found

`gpu_available()` returns `FALSE`, or `bandicoot_config()` shows OpenCL enabled
while nothing runs.

A binary install -- from CRAN, r-universe, or any other prebuilt repository --
ships the ICD loader and no device whatsoever, so it enumerates nothing until
you install a runtime yourself. Depending on the platform:

- **Linux**: PoCL (`pocl-opencl-icd` on Debian/Ubuntu) gives a CPU device that
  works well. For real GPUs install the vendor's ICD: `nvidia-opencl-icd`,
  `intel-opencl-icd`, or AMD's ROCm OpenCL runtime.
- **macOS**: Apple's OpenCL framework is always present and does enumerate the
  integrated GPU, but it has been deprecated since 10.14, is capped at OpenCL
  1.2, and reports no double-precision support (`cl_khr_fp64`) on Apple
  silicon -- so `gpu_element_square()`, the package's only double-precision
  entry point, will not run on it. PoCL from Homebrew (`brew install pocl`)
  adds a CPU device that does support double precision, with the caveat below.
- **Windows**: install the GPU vendor's driver. Intel's `oclcpuexp` CPU runtime
  enumerates a device, with the caveat below.

### `Cannot open required kernel source.`

Bandicoot loads its `.cl` kernel sources from disk at run time and could not find
the directory they live in.

From a downstream package, this almost always means the `src/Makevars` above is
missing, so `COOT_KERNEL_SOURCE_DIR` was never defined. From RcppBandicoot
itself, compare `RcppBandicoot:::get_kernel_source_dir()` against
`system.file("include", "bandicoot_bits", "ks", package = "RcppBandicoot")`:
they should be the same path with a trailing slash, and it should contain an
`opencl/` subdirectory. If the second is `""`, the kernel sources did not
survive installation and the package needs to be reinstalled.

### The R session segfaults

The session dies outright -- no R error, no traceback -- during a GPU operation.
These are bugs in the OpenCL runtimes, not in RcppBandicoot or Bandicoot, and
the crash happens inside the runtime's kernel compiler where R cannot catch it.

- **macOS with PoCL**: PoCL's LLVM JIT crashes nondeterministically, on a
  different operation each run, which is the signature of a code-generation bug
  rather than one broken operation. Setting `POCL_WORK_GROUP_METHOD=cbs` (which
  skips LLVM's loop vectorizer) and `POCL_CPU_MAX_CU_COUNT=1` makes it much less
  frequent, but does not eliminate it.
- **Windows with Intel's `oclcpuexp` runtime**: every compute kernel crashes.
  This was established with a probe that ran each operation in its own R process:
  the device queries succeed, and then all of them die, down to `gpu_eye(3)`.
  There is no known workaround.

Linux, on PoCL or a vendor runtime, is currently the only platform that runs the
kernels reliably. The macOS and Windows continuous integration legs are marked
experimental for exactly this reason: they show that the package compiles,
installs and passes its device-free tests, and nothing more.

## Status

The package is under active development with releases to [CRAN](https://cran.r-project.org/)
about once a month.

## Requirements

- **C++14 compatible compiler** (required by Bandicoot library)
- OpenCL 1.2 or later, or CUDA, for GPU support
- A vendor OpenCL runtime registering at least one device (see above)
- CLBlast (recommended) or clBLAS for OpenCL BLAS operations

### Authors

James Joseph Balamuta and Dirk Eddelbuettel

### License

GPL (>= 2)
