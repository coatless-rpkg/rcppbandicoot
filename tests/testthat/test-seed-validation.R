# gpu_set_seed() argument validation.
#
# Device-free, so it stays in the core suite (tests/testthat.R) rather than
# moving to the per-operation runners: rejecting a bad seed happens entirely in
# R, before the runtime is touched, and must keep being checked on machines and
# CRAN flavours with no device at all.  The device-backed seeding behaviour is
# in test-gpu-op-seed.R.

test_that("gpu_set_seed() rejects values that cannot be a seed", {
  # Device-free: validation happens before the runtime is touched.
  expect_error(gpu_set_seed(-1), "non-negative")
  expect_error(gpu_set_seed(1.5), "whole number")
  expect_error(gpu_set_seed(NA_real_), "whole number")
  expect_error(gpu_set_seed(Inf), "finite")
})
