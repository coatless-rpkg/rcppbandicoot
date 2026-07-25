#include <RcppBandicoot.h>

#include <sstream>
#include <string>

//' Version of the vendored Bandicoot library
//'
//' Reports the version of the Bandicoot C++ headers this package was compiled
//' against. The value is a compile-time constant, so no GPU device is touched
//' and the runtime is not initialised.
//'
//' @return
//' A length-one character vector holding the version as `"major.minor.patch"`.
//' @export
//' @examples
//' bandicoot_version()
//' package_version(bandicoot_version()) >= "4.0.0"
// [[Rcpp::export]]
std::string bandicoot_version() {
  // Assembled from the three version macros rather than taken from
  // coot::coot_version::as_string(), which appends the upstream release name
  // ("4.0.2 (Bandwidth Glutton)").  A bare "major.minor.patch" is what
  // package_version() and inst/version.txt both need; the release name would
  // make package_version() throw.
  std::stringstream ss;
  ss << coot::coot_version::major << '.'
     << coot::coot_version::minor << '.'
     << coot::coot_version::patch;

  return ss.str();
}

//' Initialise the Bandicoot GPU runtime
//'
//' Selects and initialises a device for the backend this package was built
//' against. Bandicoot also initialises lazily on the first GPU call, so this
//' is only needed when you want the device chosen (and reported) up front.
//'
//' @param print_info Logical. Print the selected device's properties to the
//'   console. Default `FALSE`.
//' @return
//' `TRUE` if a device was selected, `FALSE` otherwise. Called for its side
//' effect of initialising the runtime.
//' @export
//' @examples
//' # Not run: coot_init() releases and rebuilds the runtime's context on every
//' # call without evicting the compiled-kernel cache, so calling any of
//' # gpu_initialize(), gpu_available() or gpu_device_info() more than once in a
//' # session leaves dangling kernel handles and every later GPU call fails with
//' # cl_invalid_context. R CMD check runs all examples in ONE process, so only
//' # gpu_available() is executed there; see its help page.
//' \dontrun{
//' gpu_initialize()
//' }
// [[Rcpp::export]]
bool gpu_initialize(bool print_info = false) {
  // Deliberately the single-argument overload.  coot_init(const char*, bool,
  // uword, uword) passes its arguments to init_rt() in the wrong order
  // (coot_init.hpp:64 vs coot_rt_bones.hpp:661), which forces manual device
  // selection at platform 0 / device 0 and skips the priority-based search.
  // The one-argument form reaches init_rt(print_info) and does the right thing
  // with the backend configure baked in as COOT_DEFAULT_BACKEND.
  return coot::coot_init(print_info);
}

//' Is a usable GPU device available?
//'
//' Runs a one-element GPU operation and reports whether it succeeded. Unlike
//' [bandicoot_config()], which reports only what was detected at build time,
//' this answers the runtime question: is there a device, and can this build
//' actually reach its kernel sources?
//'
//' @return
//' `TRUE` if a device is present and a kernel executed, `FALSE` otherwise.
//' Never throws.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_available()
// [[Rcpp::export]]
bool gpu_available() {
  try {
    if (!coot::coot_init(false)) { return false; }
  } catch (...) {
    return false;
  }

  // Successful initialisation is necessary but not sufficient: a broken
  // COOT_KERNEL_SOURCE_DIR only surfaces when a kernel is actually compiled.
  // eye(1, 1) is the cheapest call that compiles one; eye(0, 0) is not, since
  // Mat<eT>::eye() returns early on n_elem == 0 without touching the runtime.
  try {
    coot::fmat probe = coot::eye<coot::fmat>(1, 1);
    return (probe.n_elem == 1);
  } catch (...) {
    return false;
  }
}

//' Report the selected GPU device's properties
//'
//' @return
//' A list with elements `available`, `backend` (`"opencl"`, `"cuda"` or
//' `"none"`), `fp64`, `fp16`, `subgroups`, `subgroup_size`, `n_units` and
//' `max_wg`. Never throws; on a machine with no device every capability is
//' reported as `FALSE`/zero.
//' @export
//' @examples
//' # Not run for the same single-initialisation reason given in
//' # [gpu_initialize()]: this is the one example process, and gpu_available()
//' # already spends the session's single permitted runtime initialisation.
//' \dontrun{
//' str(gpu_device_info())
//' }
// [[Rcpp::export]]
Rcpp::List gpu_device_info() {
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("available")     = false,
    Rcpp::Named("backend")       = "none",
    Rcpp::Named("fp64")          = false,
    Rcpp::Named("fp16")          = false,
    Rcpp::Named("subgroups")     = false,
    Rcpp::Named("subgroup_size") = 0.0,
    Rcpp::Named("n_units")       = 0.0,
    Rcpp::Named("max_wg")        = 0.0
  );

  bool ok = false;
  try { ok = coot::coot_init(false); } catch (...) { ok = false; }
  if (!ok) { return out; }

  #if defined(COOT_USE_OPENCL)
  if (coot::get_rt_internal().backend == coot::CL_BACKEND) {
    coot::opencl::runtime_t& rt = coot::get_rt_internal().cl_rt;
    out["available"]     = rt.is_valid();
    out["backend"]       = "opencl";
    out["fp64"]          = rt.has_float64();
    out["fp16"]          = rt.has_float16();
    out["subgroups"]     = rt.has_subgroups();
    out["subgroup_size"] = (double) rt.get_subgroup_size();
    out["n_units"]       = (double) rt.get_n_units();
    out["max_wg"]        = (double) rt.get_max_wg();
    return out;
  }
  #endif

  #if defined(COOT_USE_CUDA)
  if (coot::get_rt_internal().backend == coot::CUDA_BACKEND) {
    out["available"] = true;
    out["backend"]   = "cuda";
    out["fp64"]      = true;
    return out;
  }
  #endif

  return out;
}
