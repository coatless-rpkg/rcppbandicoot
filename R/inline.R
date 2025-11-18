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
  plugin <- Rcpp::Rcpp.plugin.maker(
    include.before = "#include <RcppBandicoot.h>",
    libs = RcppBandicootLdFlags(),
    package = "RcppBandicoot"
  )

  settings <- plugin()
  settings$env$PKG_CXXFLAGS <- paste(
    RcppBandicootCxxFlags(),
    settings$env$PKG_CXXFLAGS
  )
  settings$env$USE_CXX14 <- "yes"

  settings
}
