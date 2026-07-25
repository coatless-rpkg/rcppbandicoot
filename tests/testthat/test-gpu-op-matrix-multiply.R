# One operation per file, run by tests/zz-gpu-matrix-multiply.R in its own R
# process.  A GPU fault here is an access violation, not an R condition, so it
# would take the whole process down; isolation keeps it from destroying every
# other result.

test_that("gpu_matrix_multiply() matches a known 2x2 product", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_matrix_multiply", "")

  A <- matrix(c(1, 2, 3, 4), 2, 2)
  B <- matrix(c(5, 6, 7, 8), 2, 2)
  out <- gpu_matrix_multiply(A, B)

  expect_identical(dim(out), c(2L, 2L))
  expect_equal(out, A %*% B, tolerance = tol_f32_elementwise)
  expect_equal(out, matrix(c(23, 34, 31, 46), 2, 2),
               tolerance = tol_f32_elementwise)
})

test_that("gpu_matrix_multiply() handles non-square operands", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_matrix_multiply", "")

  A <- matrix(as.numeric(1:6), 2, 3)
  B <- matrix(as.numeric(1:6), 3, 2)
  out <- gpu_matrix_multiply(A, B)

  expect_identical(dim(out), c(2L, 2L))
  expect_equal(out, A %*% B, tolerance = tol_f32_elementwise)
})

test_that("gpu_matrix_multiply() matches R on a larger random product", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_matrix_multiply", "")

  set.seed(42)
  A <- matrix(rnorm(60 * 80), 60, 80)
  B <- matrix(rnorm(80 * 40), 80, 40)
  out <- gpu_matrix_multiply(A, B)

  # A status-only check would pass on a configuration that silently computes
  # zeros: with SDKROOT unset on macOS, CLBlastSgemm returns success and leaves
  # the output buffer zeroed.  Assert the values, and assert they are not all
  # zero, explicitly.
  expect_gt(max(abs(out)), 0)
  expect_equal(out, A %*% B, tolerance = tol_f32_reduction)
})

test_that("gpu_matrix_multiply() rejects non-conformable operands", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_matrix_multiply", "")

  expect_error(gpu_matrix_multiply(matrix(1, 2, 2), matrix(1, 3, 3)))
})
