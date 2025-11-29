#include <RcppBandicoot.h>

// [[Rcpp::export]]
void gpu_initialize(std::string type = "opencl", bool print_info = true) {
  coot::coot_init(type.c_str(), print_info);
}
