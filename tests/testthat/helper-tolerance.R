# Every exported gpu_* function except gpu_element_square operates on
# coot::fmat (binary32) and is wrapped back into an R double, so every
# comparison against an R reference is fp32-vs-fp64.  fp32 eps is 2^-23 =
# 1.19e-7, roughly 8x looser than testthat 3's default
# sqrt(.Machine$double.eps) = 1.49e-8 -- the default *will* fail, so it is
# never used on a float path.
#
# waldo compares a mean relative difference (not an element-wise maximum),
# which is why these numbers are small and stable.  Measured against a naive
# fp32 GEMM with N(0,1) inputs: 5.3e-8 at n = 2, 6.3e-8 at n = 10, 1.6e-7 at
# n = 100, 5.0e-7 at n = 1000.

# No accumulation: transpose, elementwise add, eye.  Exact for the
# exactly-representable inputs used below; 1e-6 is headroom, not necessity.
tol_f32_elementwise <- 1e-6

# One level of accumulation: GEMM, accu, mean, for n <= 1000.  ~20x headroom
# over the worst measured error, still four orders of magnitude tighter than
# any real regression, which is O(1).
tol_f32_reduction <- 1e-5

# Multi-pass tree reduction over ~1.6e5 fp32 terms.  A random-walk error model
# gives sqrt(n) * eps = 400 * 1.19e-7 = 4.8e-5; 1e-4 clears that with margin.
tol_f32_reduction_large <- 1e-4

# gpu_element_square is the only fp64 entry point; the testthat default is
# appropriate there.
tol_f64 <- testthat::testthat_tolerance()
