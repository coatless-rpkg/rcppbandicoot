# One operation per file, run by tests/zz-gpu-transpose.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.

test_that("gpu_transpose() transposes a non-square matrix", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_transpose", "")

  # Column-major: A is [1 3 5; 2 4 6].  A square matrix would not distinguish
  # a correct transpose from a layout mix-up between as<> and wrap().
  A <- matrix(as.numeric(1:6), nrow = 2, ncol = 3)
  out <- gpu_transpose(A)

  expect_identical(dim(out), c(3L, 2L))
  expect_equal(out, t(A), tolerance = tol_f32_elementwise)
  expect_equal(gpu_transpose(out), A, tolerance = tol_f32_elementwise)
})
