# One operation per file, run by tests/zz-gpu-element-square.R in its own R
# process.  A GPU fault here is an access violation, not an R condition, so it
# would take the whole process down; isolation keeps it from destroying every
# other result.

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
