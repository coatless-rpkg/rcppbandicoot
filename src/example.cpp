#include <RcppBandicoot.h>

//' Get Bandicoot version
//'
//' Prints the current version of the Bandicoot library.
//' @export
//' @examples
//' bandicoot_version()
// [[Rcpp::export]]
void bandicoot_version() {
  Rcpp::Rcout << coot::coot_version::as_string() << std::endl;
}