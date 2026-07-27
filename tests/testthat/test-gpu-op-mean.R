# One operation per file, run by tests/zz-gpu-mean.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.

test_that("gpu_mean() computes the grand mean", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_mean", "")

  # mean(mean(A)) is mean-of-column-means, which is the grand mean for any
  # rectangular matrix, and additionally exercises the Row<float> intermediate.
  A <- matrix(as.numeric(1:6), 2, 3)
  expect_equal(gpu_mean(A), 3.5, tolerance = tol_f32_reduction)
  expect_equal(gpu_mean(c(1, 2, 3)), 2, tolerance = tol_f32_reduction)

  set.seed(19)
  B <- matrix(rnorm(40 * 25), 40, 25)
  expect_equal(gpu_mean(B), mean(B), tolerance = tol_f32_reduction)
})

# The tallest matrix the tests above use is 40 rows, which is exactly why the
# help page was able to promise a relative error "around 1e-7" for years: at 40
# rows that is true, and nothing here ever asked for more.  It is not true at
# 1e5 rows, where the measured error is ~1.2e-5.
#
# gpu_mean() is mean-of-column-means, and the column stage is a serial loop over
# the rows (mean_colwise_conv_pre, from reduce_colwise.cl), so accuracy is
# governed by the reduced dimension.  The two tests below cover the same 1e5
# values in both shapes, each against the tolerance the model in
# helper-tolerance.R gives for its own row count.
#
# Both are upper bounds, deliberately: they will not fail if some device reduces
# columns by a tree and comes back more accurate than the model allows.  Pinning
# the degradation itself would mean asserting that an implementation is bad at
# arithmetic, and the honest version of that assertion -- the worst case over 30
# draws here was only 3.6x the old 1e-7 claim -- is too thin to be anything but
# flaky on hardware nobody has measured.

test_that("gpu_mean() holds the modelled accuracy on a tall matrix", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_mean", "")

  set.seed(23)
  n_rows <- 1e5
  A <- matrix(runif(n_rows), n_rows, 1)

  expect_equal(gpu_mean(A), mean(A), tolerance = tol_f32_reduction_n(n_rows))
})

test_that("gpu_mean() accuracy tracks the row count, not the element count", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_mean", "")

  # The same 1e5 values as the test above, reshaped.  Ten rows means the serial
  # per-column loop is ten iterations long however many columns there are, so
  # this one has to stay inside the flat small-n tolerance that the 1e5-row
  # matrix is explicitly not held to.  That is the whole claim: the reduced
  # dimension is what costs accuracy, not the amount of data.
  set.seed(23)
  wide <- matrix(runif(1e5), 10, 1e4)

  expect_equal(gpu_mean(wide), mean(wide), tolerance = tol_f32_reduction)
})
