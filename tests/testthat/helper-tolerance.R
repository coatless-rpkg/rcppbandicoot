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

# A reduction over n fp32 terms, where n is large enough that the flat number
# above no longer holds.  Same random-walk error model as the note at the top:
# n roundings of size eps, each independent of the last, accumulate as
# sqrt(n) * eps.  The factor of 2 is headroom on the model, not a number fitted
# to any measurement.
f32_eps <- 2^-23  # 1.19e-7

tol_f32_reduction_n <- function(n) 2 * sqrt(n) * f32_eps

# Multi-pass tree reduction over ~1.6e5 fp32 terms.  Written as the model
# rather than as the 1e-4 it used to be spelled: tol_f32_reduction_n(1.6e5) is
# 9.5e-5, so the two agree, and the model now lives in exactly one place.
tol_f32_reduction_large <- tol_f32_reduction_n(400 * 400)

# The model is not decoration.  gpu_mean() reduces each column with a serial
# per-column kernel (mean_colwise_conv_pre, generated from reduce_colwise.cl:
# one work item per column running a loop over the rows), not a tree, so its
# error tracks the ROW count and keeps growing with it.  Measured on an Apple
# OpenCL device, worst of 20 runif() draws, as a fraction of sqrt(n_rows) * eps:
# 0.29 at 1e3 rows, 0.25 at 1e4, 0.31 at 1e5, 0.14 at 1e6, 0.19 at 1e7.  The
# model bounds the observation across four orders of magnitude, which is what
# makes it usable as a tolerance rather than a curve fit.
#
# gpu_sum() reduces with a tree and stayed at or below 1.0e-7 -- under one eps
# -- over that same range, so it needs no size-dependent term.

# gpu_element_square is the only fp64 entry point; the testthat default is
# appropriate there.
tol_f64 <- testthat::testthat_tolerance()
