# RcppBandicoot 4.0.0.1

- Upgraded to bandicoot 4.0.0: "Bandwidth Glutton" (2026-04-24)
  - Major overhaul of the internal meta-programming framework, allowing
    automatic fusion of many operations into a single GPU operation.
  - Preliminary, experimental support for the Vulkan backend.
  - New `.is_symmetric()` and `.is_sympd()` member functions.
  - Bug fixes for in-place matrix multiplication and backend memory safety.
  - Various optimisations for better expression handling.

# RcppBandicoot 3.1.0.1

- Upgraded to bandicoot 3.1.0: "This Coffee Is Poorly Roasted" (2025-12-10)
  - Non-contiguous submatrix views: `Mat.elem(v)`, `Mat.submat(rows, cols)`,
    `Cube.elem(v)`, `Mat.rows(rows)`, `Mat.cols(cols)`.
  - Use NVRTC PTX compilation when the card architecture is too new for the
    installed NVRTC version.
  - Install into `/usr/local/` by default on Linux/macOS systems.
  - New `COOT_TARGET_OPENCL_VERSION` configuration macro to suppress OpenCL
    compilation warnings.
  - New `coot_backend()` to report the selected backend.
  - Bug fix for operations on mixed matrix or cube types.
- Kernel sources are now installed under `inst/include/bandicoot_bits/ks/`
  (renamed from `kernels/`) to avoid R CMD check path-length warnings
  ([#26](https://github.com/coatless-rpkg/rcppbandicoot/pull/26)).
- Package updates following upstream 3.1.0: configure and Makevars refresh,
  `COOT_KERNEL_SOURCE_DIR` now appends `kernels/`, and an OpenCL file
  computation correction that mirrors upstream MR 193
  ([#17](https://github.com/coatless-rpkg/rcppbandicoot/pull/17)).
- Suggest a sensible default for `COOT_TARGET_OPENCL_VERSION` during
  configuration
  ([#11](https://github.com/coatless-rpkg/rcppbandicoot/pull/11), closes
  [#9](https://github.com/coatless-rpkg/rcppbandicoot/issues/9)).

# RcppBandicoot 3.0.1.1

- Upgraded to bandicoot 3.0.1: "Extreme Cable Organization" (2025-11-19)
  - Fix for missing `cl_half.h` header on macOS.
  - Fixes for compilation warnings on gcc and clang.
  - New `COOT_KERNEL_SOURCE_DIR` macro to allow kernels to live in a custom
    location for downstream packaging.

# RcppBandicoot 3.0.0.1

- Upgraded to bandicoot 3.0.0: "Extreme Cable Organization" (2025-11-14)
  - Initial support for the `fp16` half-precision type.
  - Optimisations and simplifications for on-demand kernel compilation.
  - Fix backend initialization with CUDA toolkit version 13.
  - Support for [CLBlast](https://cnugteren.github.io/clblast/clblast.html)
    as a BLAS implementation for the OpenCL backend.
  - Fixes for OpenCL+OpenMP compilation.
  - Bug fixes for warnings in OpenCL and CUDA kernels.
  - Kernels are now compiled on-demand instead of all at once.
  - Support for `char`/`short` matrices and convenience `u8`/`s8`/`u16`/`s16`
    typedefs.
  - Standalone `replace()` function.
  - `.replace()` member function for matrices and cubes.

# RcppBandicoot 2.1.1.1

- Upgraded to bandicoot 2.1.1: "Flat Tire" (2025-05-03), which resolves a
  minor release issue in 2.1.0. Incoming upstream changes (from 2.1.0) include:
  - `.is_finite()`, `.has_inf()`, and `.has_nan()` member functions for
    matrices and cubes.
  - `.copy_size()` member function for matrices and cubes.
  - `min()`, `max()`, `index_min()`, and `index_max()` for cubes.
  - Constructors for `Mat`, `Col`, and `Row` that accept strings and
    `std::vector`s.
  - Element initialisation that handles nested initialiser lists.
  - Bug fix for copy and move operators for `Cube` and `Mat` aliases.
  - Bug fix for `diagmat()` matrix multiplication on subviews.

# RcppBandicoot 2.0.0.1

- Upgraded to bandicoot 2.0.0: "Pollen River" (2025-04-15)
  - New `Cube` class and basic functionality.
  - New `.each_row()` and `.each_col()` member functions.
  - New `regspace()` for generating vectors with regularly spaced elements.
  - Better support for Armadillo objects in `conv_to`.

# RcppBandicoot 1.16.2.1

- Initial embedded release of the [bandicoot](https://coot.sourceforge.io/)
  C++ GPU linear algebra library (by the Armadillo team) at version 1.16.2:
  "Printable Plastic Profusion" (2025-01-27).
  - Fix linking issues when the system OpenCL version does not match the
    version used when compiling the Bandicoot library.
- Provides R bindings (`Rcpp`-based) and header-only inclusion via `LinkingTo`
  for downstream packages targeting OpenCL or CUDA backends.
