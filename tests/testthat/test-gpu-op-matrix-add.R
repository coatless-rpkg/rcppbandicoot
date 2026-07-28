# One operation per file, run by tests/zz-gpu-matrix-add.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.

test_that("gpu_matrix_add() adds elementwise", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_matrix_add", "")

  A <- matrix(c(1, 2, 3, 4), 2, 2)
  B <- matrix(c(5, 6, 7, 8), 2, 2)
  expect_equal(gpu_matrix_add(A, B), A + B, tolerance = tol_f32_elementwise)

  set.seed(3)
  C <- matrix(rnorm(2 * 5), 2, 5)
  D <- matrix(rnorm(2 * 5), 2, 5)
  expect_equal(gpu_matrix_add(C, D), C + D, tolerance = tol_f32_elementwise)
})
