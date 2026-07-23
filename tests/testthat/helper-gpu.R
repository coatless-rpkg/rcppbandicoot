# Custom skip helpers live here, not in R/, per testthat's Skipping vignette:
# they are auto-sourced by test_check() and never pollute the namespace.

env_true <- function(name) {
  isTRUE(as.logical(Sys.getenv(name, "false")))
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
  if (isTRUE(RcppBandicoot::gpu_available())) {
    return(invisible(TRUE))
  }
  if (env_true("RCPPBANDICOOT_REQUIRE_GPU")) {
    info <- paste(utils::capture.output(utils::str(RcppBandicoot::gpu_device_info())),
                  collapse = " ")
    stop("RCPPBANDICOOT_REQUIRE_GPU is set but gpu_available() returned FALSE. ",
         "Device info: ", info, call. = FALSE)
  }
  testthat::skip("no usable OpenCL/CUDA device available")
}

# gpu_element_square is the package's only double-precision entry point.
# Asked directly of the runtime rather than by catching an error from
# gpu_element_square itself, which currently fails for an unrelated upstream
# reason (see test-gpu-elementwise.R).
skip_if_no_fp64 <- function() {
  skip_if_no_gpu()
  if (isTRUE(RcppBandicoot::gpu_device_info()$fp64)) {
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
