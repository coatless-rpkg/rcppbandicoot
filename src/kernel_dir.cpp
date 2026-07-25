// Runtime resolution of Bandicoot's kernel source directory.
//
// See the long comment in inst/include/RcppBandicootForward.h for why the
// COOT_KERNEL_SOURCE_DIR macro is a call to rcppbandicoot_kernel_dir() rather
// than a compile-time string literal: it lets the kernels load from wherever
// the package actually landed at run time, which a baked absolute path cannot
// do for a relocated or binary-installed package.

#include <string>
#include <Rcpp.h>

// Single storage for the kernel source directory, shared by the getter below
// (called from inside the bandicoot headers) and the setter exported to R.
// It is a function-local static so there is exactly one instance across every
// translation unit that includes the bandicoot headers.
//
// It is initialised to the path baked at configure time, which is correct for
// a source install on this machine, so the getter never returns an empty path
// even before .onLoad runs. R's .onLoad overrides it with the system.file()
// path, which is correct for any install layout, source or binary.
static std::string& kernel_dir_ref()
  {
  static std::string dir(
    #ifdef RCPPBANDICOOT_KERNEL_DIR_DEFAULT
      RCPPBANDICOOT_KERNEL_DIR_DEFAULT
    #else
      ""
    #endif
  );
  return dir;
  }

// Called from within the bandicoot headers via the COOT_KERNEL_SOURCE_DIR
// macro. The returned pointer is valid for the duration of the call; bandicoot
// copies it into a std::string immediately (std::string(source_dir) + ...).
//
// Single-writer by design: .onLoad sets the path exactly once at package load,
// on R's single main thread, before any exported function can run, so no read
// here ever races a write.
const char* rcppbandicoot_kernel_dir()
  {
  return kernel_dir_ref().c_str();
  }

//' Set the Bandicoot kernel source directory
//'
//' Internal. Called from \code{.onLoad} with the installed location of the
//' bandicoot kernel sources so they resolve at run time regardless of where
//' the package was built. The trailing slash is required: Bandicoot appends
//' \code{"opencl/<file>"} to this path.
//'
//' @param path Directory holding the bandicoot kernel sources, with a
//'   trailing slash.
//' @return Invisibly \code{NULL}.
//' @keywords internal
// [[Rcpp::export]]
void set_kernel_source_dir(std::string path)
  {
  kernel_dir_ref() = path;
  }

//' Get the Bandicoot kernel source directory
//'
//' Internal. Reports the path the compiled code will actually hand to
//' Bandicoot when it loads a kernel, which is the value \code{.onLoad} last
//' wrote. It exists so the tests can assert on the compiled state directly:
//' checking an independently recomputed \code{system.file()} path proves
//' nothing about what the shared object holds, so a \code{.onLoad} regression
//' would leave every GPU test skipping and the suite green.
//'
//' @return
//' A length-one character vector holding the directory, with a trailing slash.
//' Empty only if the package was built without a configure-time default and
//' \code{.onLoad} has not run.
//' @keywords internal
// [[Rcpp::export]]
std::string get_kernel_source_dir()
  {
  return kernel_dir_ref();
  }
