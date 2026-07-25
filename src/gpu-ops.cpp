#include <RcppBandicoot.h>

#include <cmath>

//' GPU Matrix Multiplication
//'
//' Multiply two matrices on the GPU using Bandicoot
//'
//' @param A First matrix
//' @param B Second matrix
//' @return Product of A and B computed on GPU
//' @section Precision:
//' The product is computed in single precision (`float`). R's doubles are
//' rounded on the way to the device and the result is widened back, so
//' expect a relative error around `1e-7` against `A %*% B` in base R, growing
//' with the shared inner dimension as the rounding accumulates.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' A <- matrix(c(1, 2, 3, 4), 2, 2)
//' B <- matrix(c(5, 6, 7, 8), 2, 2)
//' gpu_matrix_multiply(A, B)
// [[Rcpp::export]]
coot::fmat gpu_matrix_multiply(const coot::fmat& A, const coot::fmat& B) {
  return A * B;
}

//' GPU Matrix Transpose
//'
//' Transpose a matrix on the GPU using Bandicoot
//'
//' @param A Matrix to transpose
//' @return Transposed matrix computed on GPU
//' @section Precision:
//' The matrix is held on the device in single precision (`float`).
//' Transposition only moves elements, but the round trip through `float`
//' rounds every one of them, so entries needing more than about 7 significant
//' decimal digits do not come back bit-identical to the input.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_transpose(matrix(1:6, nrow = 2))
// [[Rcpp::export]]
coot::fmat gpu_transpose(const coot::fmat& A) {
 return coot::trans(A);
}

//' GPU Matrix Addition
//'
//' Add two matrices on the GPU using Bandicoot
//'
//' @param A First matrix
//' @param B Second matrix
//' @return Sum of A and B computed on GPU
//' @section Precision:
//' The sum is computed in single precision (`float`). R's doubles are rounded
//' on the way to the device and the result is widened back, so expect a
//' relative error around `1e-7` against `A + B` in base R.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_matrix_add(matrix(c(1, 2, 3, 4), 2, 2), matrix(c(5, 6, 7, 8), 2, 2))
// [[Rcpp::export]]
coot::fmat gpu_matrix_add(const coot::fmat& A, const coot::fmat& B) {
 return A + B;
}

//' GPU Element-wise Operations
//'
//' Apply element-wise square operation on GPU
//'
//' @param A Input matrix
//' @return Matrix with each element squared, computed on GPU
//' @section Precision:
//' This is the only operation in the package that runs in double precision
//' (`double`), so the result matches `A^2` in base R to full double accuracy.
//' It therefore needs a device with 64-bit floating point support: check
//' `gpu_device_info()$fp64` before calling, as the OpenCL backend requires the
//' `cl_khr_fp64` extension.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_element_square(matrix(c(1, 2, 3, 4), 2, 2))
// [[Rcpp::export]]
coot::mat gpu_element_square(const coot::mat& A) {
 return coot::square(A);
}

//' GPU Sum
//'
//' Calculate the sum of all elements in a matrix on GPU
//'
//' @param A Input matrix
//' @return Sum of all elements
//' @section Precision:
//' The matrix is held and accumulated in single precision (`float`) even
//' though the value handed back to R is a double, so the sum carries about 7
//' significant decimal digits rather than 16. Expect a relative error around
//' `1e-7` against `sum(A)`, growing with the number of elements accumulated.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_sum(matrix(1:6, nrow = 2))
// [[Rcpp::export]]
double gpu_sum(const coot::fmat& A) {
 return coot::accu(A);
}

//' GPU Mean
//'
//' Calculate the mean of all elements in a matrix on GPU
//'
//' @param A Input matrix
//' @return Mean of all elements
//' @section Precision:
//' The matrix is held and averaged in single precision (`float`) even though
//' the value handed back to R is a double, so the mean carries about 7
//' significant decimal digits rather than 16. Expect a relative error around
//' `1e-7` against `mean(A)`.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_mean(matrix(1:6, nrow = 2))
// [[Rcpp::export]]
double gpu_mean(const coot::fmat& A) {
 return coot::mean(coot::mean(A));
}

//' Create Identity Matrix on GPU
//'
//' Create an identity matrix on the GPU
//'
//' @param n Size of the identity matrix
//' @return n x n identity matrix on GPU
//' @section Precision:
//' The matrix is created in single precision (`float`). Zero and one are both
//' exact in `float`, so the result is exact.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_eye(3)
// [[Rcpp::export]]
coot::fmat gpu_eye(int n) {
 return coot::eye<coot::fmat>(n, n);
}

// Has the device generator been seeded in this session yet?  Shared by
// gpu_set_seed() and gpu_randu() so that an explicit gpu_set_seed() call is not
// silently overwritten by gpu_randu()'s lazy seeding on the next draw.  A
// function-local static gives exactly one instance for the lifetime of the
// process, which is the scope the seeding decision belongs to.
static bool& gpu_rng_seeded() {
  static bool seeded = false;
  return seeded;
}

//' Set the GPU random number generator seed
//'
//' Seeds the device-side generator that backs [gpu_randu()]. Bandicoot
//' defaults that seed to `0` and never randomises it, so without an explicit
//' seed every fresh process would draw the same numbers.
//'
//' @param seed Seed for the device generator: a non-negative whole number no
//'   larger than `2^53`. R has no unsigned 64-bit type, so the seed arrives as
//'   a double and is converted to the 64-bit unsigned value Bandicoot wants;
//'   `2^53` is the largest integer a double still represents exactly.
//' @return Invisibly `NULL`. Called for its side effect on the device
//'   generator.
//' @seealso [gpu_randu()], which seeds the device from R's own random number
//'   stream on the first draw of a session when this has not been called.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' gpu_set_seed(42)
//' gpu_randu(2, 2)
// [[Rcpp::export]]
void gpu_set_seed(double seed) {
  if (ISNAN(seed)) {
    Rcpp::stop("'seed' must be a non-negative whole number, not NA or NaN.");
  }
  if (!R_FINITE(seed)) {
    Rcpp::stop("'seed' must be finite.");
  }
  if (seed < 0.0) {
    Rcpp::stop("'seed' must be non-negative.");
  }
  if (seed != std::floor(seed)) {
    Rcpp::stop("'seed' must be a whole number.");
  }
  // Past 2^53 consecutive integers stop being individually representable as
  // doubles, so a larger value silently means "some nearby even number" and the
  // caller would not get the seed they asked for.
  if (seed > 9007199254740992.0) {
    Rcpp::stop("'seed' must be no larger than 2^53 (9007199254740992).");
  }

  coot::coot_rng::set_seed((coot::u64) seed);
  gpu_rng_seeded() = true;
}

// Draws a device generator seed from R's own random number stream.  Two draws
// are combined because unif_rand() yields at most 32 bits on R's default
// generator, while the device generator takes a 64-bit seed.
static coot::u64 gpu_seed_from_r_rng() {
  const double two_pow_32 = 4294967296.0;

  const coot::u64 hi = (coot::u64) (R::unif_rand() * two_pow_32);
  const coot::u64 lo = (coot::u64) (R::unif_rand() * two_pow_32);

  return (hi << 32) ^ lo;
}

//' Create Random Matrix on GPU
//'
//' Create a matrix with uniformly distributed random values on the GPU
//'
//' @param n_rows Number of rows
//' @param n_cols Number of columns
//' @return Random matrix on GPU
//' @section Seeding:
//' Bandicoot defaults the device generator's seed to `0` and never randomises
//' it, so on its own this function would return the same matrix in every fresh
//' R process. To avoid that, the *first* GPU random draw in a session takes its
//' seed from R's own random number stream; later draws continue from wherever
//' the device generator left off. Two consequences follow: `set.seed()` before
//' that first draw makes the whole session reproducible, and drawing that seed
//' consumes two values from R's stream. Call [gpu_set_seed()] to override the
//' seed explicitly instead, at any point including before the first draw.
//' @section Precision:
//' Values are generated in single precision (`float`) and widened to doubles on
//' the way back to R. The draws therefore take roughly `2^24` distinct values
//' in `[0, 1)` rather than the far finer grid `runif()` produces.
//' @export
//' @examplesIf nzchar(Sys.getenv("RCPPBANDICOOT_RUN_GPU_EXAMPLES"))
//' dim(gpu_randu(4, 3))
// [[Rcpp::export]]
coot::fmat gpu_randu(int n_rows, int n_cols) {
 // Seed once per session, lazily: doing it here rather than in .onLoad means a
 // set.seed() call issued after the package is loaded still reaches the device,
 // and means a session that never draws GPU randoms never disturbs R's stream.
 // Reading R's stream is safe because compileAttributes() emits an RNGScope for
 // this function (src/RcppExports.cpp).
 if (!gpu_rng_seeded()) {
   coot::coot_rng::set_seed(gpu_seed_from_r_rng());
   gpu_rng_seeded() = true;
 }

 return coot::randu<coot::fmat>(n_rows, n_cols);
}
