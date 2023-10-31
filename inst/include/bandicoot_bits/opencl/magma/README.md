### opencl/magma

This directory contains adaptations of the MAGMA and clMAGMA source code, to
provide GPU-capable LAPACK functionality that we have not implemented by hand
for bandicoot.

MAGMA is not available in most package managers, and clMAGMA is now
unmaintained; therefore, the best option is to include it in our source code.

MAGMA is primarily used for more complicated decompositions, such as `svd()`,
`chol()`, `eig()`, and so forth.

#### Porting new MAGMA functions

MAGMA is written as a C library, for CUDA, and many changes are necessary to
make it a single-header implementation in the format bandicoot requires.

Once you have determined that you need a new function from MAGMA (say, dgesvd),
it would be wise to look through the source of that function (usually
`magma-x.y.z/src/dgesvd.cpp` or `magma-x.y.z/src/dgesvd_gpu.cpp`) and find any
dependencies that have not yet been ported.  If there are any of those
dependencies, it's probably best to start by porting those simpler dependencies
first.

In general a series of changes are needed:

 * All functions need to be declared `inline` so they can be a part of
   bandicoot (which is header-only).

 * Pointer arithmetic for GPU arrays (`magmaDouble_ptr` or similar) are not
   allowed in OpenCL.  Instead, all GPU functions take an extra `size_t offset`
   parameter.  So, e.g., instead of `magmaDouble_ptr dA`, the parameters will be
   `magmaDouble_ptr dA, const size_t dA_offset`.  You will need to update that
   in the signature of whatever function you are porting (if it takes GPU
   arrays), and in any calls to other functions that involve a GPU array.

 * The awful `#define A(i, j) A[i + j * lda]`-style macros need to be removed.
   Go through the source code and manually expand the macros yourself.  Yes,
   it's tedious...

 * All LAPACK calls, like `lapackf77_dgesvd(...)` must be replaced with
   `lapack::gesvd(...)`.  If, at compile time, you find that
   particular CPU LAPACK function doesn't exist, add it to
   `bandicoot_bits/translate_lapack.hpp` (or `bandicoot_bits/translate_blas.hpp`
   for BLAS functions), as well as `bandicoot_bits/def_lapack.hpp` (or
   `bandicoot_bits/def_blas.hpp` for BLAS functions).  Note that the `lapack::`
   and `blas::` provided functions don't take pointers for input arguments and
   so the calls may need to be adjusted somewhat.

 * Replace anything that creates a `magma_queue_t` and populates it with the
   simpler call `magma_queue_t queue = magma_queue_create()` (or similar).

 * If you are an overachiever, you can match the style to the rest of the
   bandicoot codebase.

Once that is done, make sure to port the test (e.g.,
`magma-x.y.z/testing/testing_dgesvd.cpp`) into tests/magma/.  There are lots of
things in the tests that can be stripped out; for instance, there is no need for
a `magma_opts` struct.  You can just hard-code the defaults for those tests.

Don't hesitate to look at existing implementations of functions and compare them
side-by-side with the original MAGMA or clMAGMA implementations to get a better
idea of what needs to be done.  The same can also be done for the tests.
