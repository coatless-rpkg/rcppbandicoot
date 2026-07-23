test_that("gpu_eye() produces an identity matrix", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_eye", "")

  out <- gpu_eye(3)
  expect_identical(dim(out), c(3L, 3L))
  expect_equal(out, diag(3), tolerance = tol_f32_elementwise)
  expect_equal(gpu_eye(1), matrix(1), tolerance = tol_f32_elementwise)
})

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

test_that("gpu_element_square() squares elementwise in double precision", {
  skip_if_no_fp64()
  # Bandicoot 4.0.2 emits `cx_double` (and out-of-range float literals) into
  # the generated fp64 kernel source without the prelude that defines it.
  # Reproduced on both PoCL and Apple's own OpenCL compiler, so it is an
  # upstream kernel-generation defect, not a driver problem.
  skip_if_xfail("gpu_element_square",
                "bandicoot 4.0.2 emits cx_double into fp64 kernel source")

  A <- matrix(c(1, 2, 3, 4), 2, 2)
  expect_equal(gpu_element_square(A), A^2, tolerance = tol_f64)

  set.seed(11)
  B <- matrix(rnorm(50), 10, 5)
  expect_equal(gpu_element_square(B), B^2, tolerance = tol_f64)
})
