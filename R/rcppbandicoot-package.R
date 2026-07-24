#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib RcppBandicoot, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    ## Point Bandicoot at the kernel sources as they actually landed on THIS
    ## machine. The path is otherwise baked in at build time, which is wrong
    ## for a binary package installed somewhere other than it was built (as
    ## CRAN ships macOS and Windows builds). system.file() resolves to the real
    ## install location; the trailing slash is required because Bandicoot
    ## appends "opencl/<file>". See src/kernel_dir.cpp and RcppBandicootForward.h.
    ks <- system.file("include", "bandicoot_bits", "ks", package = pkgname)
    if (nzchar(ks)) {
        set_kernel_source_dir(paste0(ks, "/"))
    }
}
