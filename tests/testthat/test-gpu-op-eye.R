# One operation per file, run by tests/zz-gpu-eye.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.

test_that("gpu_eye() produces an identity matrix", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_eye", "")

  out <- gpu_eye(3)
  expect_identical(dim(out), c(3L, 3L))
  expect_equal(out, diag(3), tolerance = tol_f32_elementwise)
  expect_equal(gpu_eye(1), matrix(1), tolerance = tol_f32_elementwise)
})
