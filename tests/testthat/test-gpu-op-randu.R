# One operation per file, run by tests/zz-gpu-randu.R in its own R process.
# A GPU fault here is an access violation, not an R condition, so it would take
# the whole process down; isolation keeps it from destroying every other result.

test_that("gpu_randu() returns the requested shape within [0, 1]", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_randu", "")

  # Values are never asserted: Bandicoot's RNG is seeded with 0 at device-init
  # time, its stream length depends on the device's work-group size, and it is
  # not connected to R's set.seed().  Shape, range and non-degeneracy are all
  # that can portably be checked.
  X <- gpu_randu(64, 32)
  expect_identical(dim(X), c(64L, 32L))
  expect_true(all(is.finite(X)))
  expect_true(all(X >= 0))
  expect_true(all(X <= 1))
  # Guards against a zeroed or constant buffer.
  expect_gt(length(unique(as.vector(X))), 1000L)
})

test_that("gpu_randu() is approximately uniform in the large", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_randu", "")

  X <- gpu_randu(200, 200)
  expect_equal(mean(X), 0.5, tolerance = 0.02)
  expect_equal(stats::var(as.vector(X)), 1 / 12, tolerance = 0.05)
})
