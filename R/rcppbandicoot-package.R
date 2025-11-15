#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib RcppBandicoot, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
  # Set the kernel path for Bandicoot OpenCL kernels
  kernel_path <- system.file("include/bandicoot_bits/opencl/kernels",
                            package = "RcppBandicoot")

  if (nzchar(kernel_path)) {
    Sys.setenv(COOT_CL_KERNEL_PATH = kernel_path)
  }
}
