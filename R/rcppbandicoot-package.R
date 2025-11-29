#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib RcppBandicoot, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
  # Set the kernel path for Bandicoot OpenCL kernels
  #kernel_path <- system.file("include/bandicoot_bits/opencl/kernels",
  #                          package = "RcppBandicoot")

  #if (nzchar(kernel_path)) {
  #  Sys.setenv(COOT_CL_KERNEL_PATH = kernel_path)
  #}

  #cuda_path <- system.file("include/bandicoot_bits/cuda/kernels",
  #                         package = "RcppBandicoot")
  #if (nzchar(cuda_path)) {
  #  Sys.setenv(COOT_CUDA_KERNEL_PATH = kernel_path)
  #}

    #kernel_path <- system.file("include/bandicoot_bits/", package = "RcppBandicoot")
    #cat("kernel_path: ", kernel_path, "\n")
    #if (nzchar(kernel_path)) {
    #    Sys.setenv(COOT_KERNEL_SOURCE_DIR = kernel_path)
    #}

    #gpu_initialize("cuda", TRUE)
}
