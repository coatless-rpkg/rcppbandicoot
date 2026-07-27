# The device-buffer guard's boundary, tested as the predicate it is.
#
# test-alloc-limit.R exercises the guard through real calls, which is the
# honest end-to-end check but has to be opt-in: those calls only cost nothing
# while the guard WORKS, and the moment it regresses -- exactly when the test
# earns its keep -- they ask for four gigabytes and the runner dies instead of
# reporting a failure. So the arithmetic is checked here instead, through an
# internal entry point that runs the same guard over a hypothetical size. No
# device, no memory, no skip: this runs on every push, including the
# device-free leg.

test_that("the guard permits everything below 2^32 bytes", {
  # 2^30 - 1 float elements is one element under the ceiling.
  expect_true(RcppBandicoot:::check_alloc_limit(2^30 - 1, 4L))
  expect_true(RcppBandicoot:::check_alloc_limit(1, 4L))
  expect_true(RcppBandicoot:::check_alloc_limit(0, 4L))
  # 2^29 - 1 doubles is the same ceiling at 8 bytes.
  expect_true(RcppBandicoot:::check_alloc_limit(2^29 - 1, 8L))
})

test_that("the guard refuses 2^32 bytes and above", {
  pattern <- "refuses any single device buffer of 2\\^32 bytes"
  # Exactly at the limit, for both element sizes the package uses.
  expect_error(RcppBandicoot:::check_alloc_limit(2^30, 4L), pattern)
  expect_error(RcppBandicoot:::check_alloc_limit(2^29, 8L), pattern)
  expect_error(RcppBandicoot:::check_alloc_limit(2^40, 4L), pattern)
})

test_that("the boundary is exactly one element wide", {
  # A guard off by one element, or keyed to elements rather than bytes, would
  # satisfy both tests above. This is the pair that pins it: adjacent sizes on
  # either side of the ceiling must disagree, and they must disagree at a
  # DIFFERENT element count for each element size, which is what proves the
  # limit is a byte count rather than a count of elements.
  expect_true(RcppBandicoot:::check_alloc_limit(2^30 - 1, 4L))
  expect_error(RcppBandicoot:::check_alloc_limit(2^30, 4L))

  expect_true(RcppBandicoot:::check_alloc_limit(2^29 - 1, 8L))
  expect_error(RcppBandicoot:::check_alloc_limit(2^29, 8L))

  # And a double buffer at the float ceiling is refused, since it is twice the
  # bytes -- the case that fails if sizeof(T) were ever dropped from the sum.
  expect_error(RcppBandicoot:::check_alloc_limit(2^30 - 1, 8L))
})

test_that("the refusal reports the arithmetic it applied", {
  msg <- tryCatch(RcppBandicoot:::check_alloc_limit(2^30, 4L),
                  error = conditionMessage)
  expect_match(msg, "1073741824 elements at 4 bytes each is 4294967296 bytes",
               fixed = TRUE)
  expect_match(msg, "largest permitted for this element type is 1073741823 elements",
               fixed = TRUE)
})
