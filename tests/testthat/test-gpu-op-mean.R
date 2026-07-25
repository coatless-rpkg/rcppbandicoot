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
