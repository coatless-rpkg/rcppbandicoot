#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib RcppBandicoot, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    ## this function is now empty as the kernel path is a _compile-time_
    ## and not run-time issue
}
