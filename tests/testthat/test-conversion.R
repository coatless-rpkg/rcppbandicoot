# R -> Bandicoot conversion edge cases.
#
# These guard the class of bug that is worst here: a silently wrong number.
# Before NA handling was added to inst/include/RcppBandicootAs.h, R's
# NA_integer_ (which is INT_MIN) was passed through a bare static_cast, so
# gpu_sum(matrix(c(1L, NA), 1)) returned about -2.1e9 instead of a missing
# value -- with no warning, on the integer matrices the help pages use as
# their own examples.

test_that("NA in an integer matrix does not become a large negative number", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  got <- gpu_sum(matrix(c(1L, NA_integer_), nrow = 1))

  # The single-precision path cannot carry R's NA payload -- IEEE 754 keeps the
  # NaN across the narrowing to float, but not the bit pattern that tells NA
  # apart from NaN. is.na() is TRUE for both, which is the property that
  # matters; the old behaviour returned a finite, plausible-looking number.
  expect_true(is.na(got))
  expect_false(isTRUE(got < -1e9))
})

test_that("NA in a logical matrix does not become a large negative number", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  got <- gpu_sum(matrix(c(TRUE, NA), nrow = 1))
  expect_true(is.na(got))
  expect_false(isTRUE(got < -1e9))
})

test_that("NA in a double matrix propagates", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  expect_true(is.na(gpu_sum(matrix(c(1, NA_real_), nrow = 1))))
})

test_that("integer input still converts correctly when no NA is present", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  # matrix(1:6, nrow = 2) is the documented example input for several
  # functions, so the ordinary integer path must stay exact.
  expect_equal(gpu_sum(matrix(1:6, nrow = 2)), 21, tolerance = 1e-5)
})

test_that("non-finite double input is preserved rather than truncated", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  expect_true(is.infinite(gpu_sum(matrix(c(1, Inf), nrow = 1))))
})
