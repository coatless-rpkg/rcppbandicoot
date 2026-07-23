#include <RcppBandicoot.h>

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
//' # gpu_initialize() is not called directly here: see gpu_available()'s
//' # example for why the GPU runtime may be probed at most once per R
//' # session; gpu_device_info() is used for that one probe.
//' if (is.null(getOption("rcppbandicoot.ex_info"))) {
//'   options(rcppbandicoot.ex_info = gpu_device_info())
//' }
//' .rcppbandicoot_ex_info <- getOption("rcppbandicoot.ex_info")
//' isTRUE(.rcppbandicoot_ex_info$available)
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
//' @examples
//' # gpu_available() re-initialises the GPU runtime every time it is
//' # called, and Bandicoot's runtime must not be initialised more than once
//' # per R session (a second init leaves already-compiled kernels dangling
//' # and pointed at a freed context). Every example in this package
//' # therefore probes the device via gpu_device_info() at most once per
//' # session and reuses the cached result afterwards.
//' if (is.null(getOption("rcppbandicoot.ex_info"))) {
//'   options(rcppbandicoot.ex_info = gpu_device_info())
//' }
//' .rcppbandicoot_ex_info <- getOption("rcppbandicoot.ex_info")
//' isTRUE(.rcppbandicoot_ex_info$available)
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
//' # See gpu_available()'s example for why this probe is cached and reused
//' # by every other example in this package.
//' if (is.null(getOption("rcppbandicoot.ex_info"))) {
//'   options(rcppbandicoot.ex_info = gpu_device_info())
//' }
//' .rcppbandicoot_ex_info <- getOption("rcppbandicoot.ex_info")
//' str(.rcppbandicoot_ex_info)
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
