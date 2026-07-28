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
    ##
    ## system.file() returns "" as soon as any component of the path is absent,
    ## so nzchar() below is already the existence check -- a separate
    ## dir.exists() would only repeat the stat() that system.file() just did,
    ## and .onLoad runs on every library() call.
    ##
    ## The empty case means the kernel sources did not survive installation.
    ## Left silent it surfaces much later and somewhere else entirely, as
    ## Bandicoot's "Cannot open required kernel source." from the first GPU
    ## call, so say it here while the cause is still legible. Deliberately no
    ## set_kernel_source_dir("") in that branch: leaving the configure-time
    ## default in place is strictly better, since on a source install it still
    ## names a real directory. get_kernel_source_dir() reports which path won.
    ks <- system.file("include", "bandicoot_bits", "ks", package = pkgname)
    if (nzchar(ks)) {
        set_kernel_source_dir(paste0(ks, "/"))
    } else {
        warning("RcppBandicoot: the Bandicoot kernel sources are missing from ",
                "the installed package (expected include/bandicoot_bits/ks). ",
                "GPU operations will fail with \"Cannot open required kernel ",
                "source.\" until the package is reinstalled.", call. = FALSE)
    }
}
