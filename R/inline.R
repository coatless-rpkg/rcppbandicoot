##
## RcppBandicoot inline.R
##
## Copyright (C) 2023-2025 James Balamuta
##
## Licensed under GPL-2 or later
##

#' Rcpp inline plugin for RcppBandicoot
#'
#' This function provides the default Rcpp inline plugin for RcppBandicoot.
#' It uses the configuration determined at package installation time.
#'
#' @return
#' A list containing the plugin configuration
#'
#' @export
#' @examples
#' \dontrun{
#' # Use with Rcpp::sourceCpp()
#' # File: test.cpp
#' # // [[Rcpp::depends(RcppBandicoot)]]
#' #
#' # #include <RcppBandicoot.h>
#' #
#' # // [[Rcpp::export]]
#' # coot::mat gpu_multiply(const coot::mat& A, const coot::mat& B) {
#' #   return A * B;
#' # }
#' #
#' # Rcpp::sourceCpp("test.cpp")
#' }
inlineCxxPlugin <- function() {
  # include.before, not include.after: Rcpp.plugin.maker emits
  # "<include.before>\n#include <Rcpp.h>\n<include.after>", and RcppBandicoot.h
  # refuses to be included once Rcpp.h has been (the #error at the top of that
  # header). Bandicoot has to be loaded before Rcpp so RcppBandicootForward.h
  # can declare the wrap()/Exporter specialisations first, which is exactly what
  # this ordering gives; swapping to include.after breaks the compile outright.
  plugin <- Rcpp::Rcpp.plugin.maker(
    include.before = "#include <RcppBandicoot.h>",
    libs = RcppBandicootLdFlags(),
    package = "RcppBandicoot"
  )

  # Rcpp.plugin.maker populates env with PKG_LIBS and nothing else, so
  # PKG_CXXFLAGS is NULL here and paste() would leave a stray trailing space.
  # Concatenating the non-NULL parts avoids that while still deferring to Rcpp
  # should it ever start contributing compiler flags of its own.
  settings <- plugin()
  settings$env$PKG_CXXFLAGS <- paste(
    c(RcppBandicootCxxFlags(), settings$env$PKG_CXXFLAGS),
    collapse = " "
  )

  # No USE_CXX14: `R CMD config CXX14` has been defunct since R 4.5, and R's
  # default CXX is already C++17 or later, which exceeds bandicoot's needs.
  settings
}
