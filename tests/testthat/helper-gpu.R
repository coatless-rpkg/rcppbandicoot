# Custom skip helpers live here, not in R/, per testthat's Skipping vignette:
# they are auto-sourced by test_check() and never pollute the namespace.

# Recognise the truthy spellings a workflow author might reasonably write, not
# only as.logical()'s vocabulary.  as.logical("1") is NA, so a bare
# RCPPBANDICOOT_REQUIRE_GPU=1 would otherwise be silently false -- which turns
# the loud "CI promised a device and there isn't one" failure back into a quiet
# skip, the exact green-but-tested-nothing outcome this gate exists to prevent.
env_true <- function(name) {
  tolower(trimws(Sys.getenv(name, ""))) %in% c("1", "true", "yes", "on")
}

# gpu_available()/gpu_initialize()/gpu_device_info() are each a thin wrapper
# around coot::coot_init(), and coot_init() is NOT safe to call more than once
# per process. It reaches coot_rt_t::init() (coot_rt_bones.hpp) directly,
# which -- unlike the internally-used get_rt() -- never checks `initialised`
# first: every call unconditionally runs internal_cleanup()
# (opencl/runtime_meat.hpp), releasing the live cl_context/cl_command_queue,
# then builds a fresh one. internal_cleanup() does not evict the singleton
# runtime's already-compiled cl_kernel cache (that file's own comments read
# "TODO: clean up RNGs" / "TODO: go through each kernel vector"), so any
# kernel compiled before a second init call is left as a dangling handle
# bound to the now-released context. The next time that kernel type is
# needed -- including inside runtime_t::init() itself, which compiles the
# RNG kernels as part of set-up -- it fails with CL_INVALID_CONTEXT, and
# every explicit init call after the first reports no device from then on.
#
# Verified directly in a Debian/PoCL container: a fresh session's first call
# to gpu_available() returns TRUE; gpu_available() again (or gpu_initialize()
# or gpu_device_info()) immediately after returns FALSE, and gpu_eye()
# subsequently throws "coot::opencl::fill(): couldn't execute kernel:
# cl_invalid_context" -- even though the device and build are both fine.
# Actual GPU operations (gpu_eye(), gpu_transpose(), gpu_matrix_multiply(),
# ...) are unaffected by this because Bandicoot reaches the runtime through
# the guarded get_rt() for real work and never re-inits there; it is only
# gpu_available()/gpu_initialize()/gpu_device_info() themselves, called more
# than once combined, that poison the session.
#
# skip_if_no_gpu() is the first thing every GPU test in this suite calls, so
# without caching, the second test in the suite would silently see "no GPU"
# regardless of the device -- this is a correctness requirement for the
# suite, not a performance nicety.
.rcppbandicoot_gpu_probe <- new.env(parent = emptyenv())

# The sole caller, ever, of RcppBandicoot::gpu_device_info() in this test
# suite. gpu_device_info()'s own `available` field is set from
# opencl::runtime_t::is_valid(), which -- like gpu_available()'s dedicated
# eye(1,1) probe -- only becomes true after a kernel (the RNG seed kernel)
# has actually been compiled from COOT_KERNEL_SOURCE_DIR during init, so it
# is an equally rigorous "is there a usable device" signal; gpu_available()
# itself is deliberately never called here, since calling both would already
# be the second explicit init.
cached_gpu_device_info <- function() {
  if (is.null(.rcppbandicoot_gpu_probe$info)) {
    .rcppbandicoot_gpu_probe$info <- RcppBandicoot::gpu_device_info()
  }
  .rcppbandicoot_gpu_probe$info
}

# Skip the device tests when there is no device -- unless this repo's own CI
# asked for them, in which case the absence is a hard failure.
#
# Deliberately NOT skip_on_cran(): NOT_CRAN is set by devtools::check() and
# devtools::test() only.  rcmdcheck and r-lib/actions/check-r-package never set
# it, so skip_on_cran() would fire in our own CI too and we would be back to a
# green build that ran no kernels.  RCPPBANDICOOT_REQUIRE_GPU is set in
# .github/workflows/R-CMD-check.yaml and nowhere else; CRAN cannot set it.
skip_if_no_gpu <- function() {
  info <- cached_gpu_device_info()
  if (isTRUE(info$available)) {
    return(invisible(TRUE))
  }
  if (env_true("RCPPBANDICOOT_REQUIRE_GPU")) {
    info_str <- paste(utils::capture.output(utils::str(info)), collapse = " ")
    stop("RCPPBANDICOOT_REQUIRE_GPU is set but no usable GPU device was found. ",
         "Device info: ", info_str, call. = FALSE)
  }
  testthat::skip("no usable OpenCL/CUDA device available")
}

# gpu_element_square is the package's only double-precision entry point.
# Asked directly of the runtime rather than by catching an error from
# gpu_element_square itself, which currently fails for an unrelated upstream
# reason (see test-gpu-elementwise.R).
skip_if_no_fp64 <- function() {
  skip_if_no_gpu()
  if (isTRUE(cached_gpu_device_info()$fp64)) {
    return(invisible(TRUE))
  }
  if (env_true("RCPPBANDICOOT_REQUIRE_FP64")) {
    stop("RCPPBANDICOOT_REQUIRE_FP64 is set but the selected device reports ",
         "no double-precision support", call. = FALSE)
  }
  testthat::skip("device does not support double precision (cl_khr_fp64)")
}

# Operations known to be broken on a specific device/driver combination.
#
# RCPPBANDICOOT_XFAIL is a comma-separated list of gpu_* names, set per-runner
# in the workflow so that every exclusion is visible in one file.  Unset means
# "exclude nothing" -- the honest default, so a developer running the suite
# locally sees the real behaviour.  The nightly leg sets it to the empty
# string, which also excludes nothing, so the known-broken operations run for
# real there and the nightly stays red until upstream fixes them.
gpu_xfail_ops <- function() {
  raw <- Sys.getenv("RCPPBANDICOOT_XFAIL", NA_character_)
  if (is.na(raw) || !nzchar(raw)) {
    return(character())
  }
  trimws(strsplit(raw, ",", fixed = TRUE)[[1]])
}

skip_if_xfail <- function(op, reason) {
  if (op %in% gpu_xfail_ops()) {
    testthat::skip(paste0("known failure excluded via RCPPBANDICOOT_XFAIL: ",
                          op, " (", reason, ")"))
  }
  invisible(TRUE)
}
