# One operation per file, run by tests/zz-gpu-sum.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.
#
# The multi-pass reduction path is deliberately a separate file
# (test-gpu-op-sum-multipass.R): it has its own xfail key and its own failure
# mode, so it must be able to crash without taking this one with it.

test_that("gpu_sum() sums all elements", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "")

  A <- matrix(as.numeric(1:6), 2, 3)
  expect_equal(gpu_sum(A), 21, tolerance = tol_f32_reduction)
  # R vectors are converted to n x 1 matrices by the Exporter.
  expect_equal(gpu_sum(c(1, 2, 3)), 6, tolerance = tol_f32_reduction)
})
