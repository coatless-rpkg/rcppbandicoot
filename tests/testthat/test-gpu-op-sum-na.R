# R -> Bandicoot conversion edge cases: inputs that must survive the trip
# intact, and inputs that must be refused outright.
#
# These guard the class of bug that is worst here: a silently wrong number.
# Before NA handling was added to inst/include/RcppBandicootAs.h, R's
# NA_integer_ (which is INT_MIN) was passed through a bare static_cast, so
# gpu_sum(matrix(c(1L, NA), 1)) returned about -2.1e9 instead of a missing
# value -- with no warning, on the integer matrices the help pages use as
# their own examples.
#
# These run gpu_sum on the device, so they get their own R process
# (tests/zz-gpu-sum-na.R) like every other device-touching file: a GPU fault is
# an access violation, not an R condition, and would take the process down.

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

# Complex input was the same defect wearing different clothes, and it survived
# the NA fix by about thirty lines: the CPLXSXP branch of bandicoot_copy_from_r
# kept the real part and threw the imaginary one away, so
# gpu_sum(matrix(c(1+2i, 0+3i), 1, 2)) returned 1 where base R's sum() returns
# 1+5i.  A plausible number, no warning, wrong.
#
# The package exports no complex-valued operation, so refusing is the entire
# fix; these check that the refusal is a loud R error naming what went wrong and
# not, say, a coercion warning followed by the same wrong answer.  They also
# need no device -- the Exporter stops before it constructs anything -- but they
# live here rather than in the core suite because a regression would put them
# straight back on the GPU.

test_that("complex input is refused rather than reduced to its real part", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  expect_error(gpu_sum(matrix(c(1+2i, 0+3i), 1, 2)), "complex input")
  # Not 1: that is what the old real-part fallback returned, and a test that
  # only asked for "an error" would have been satisfied by a coercion warning
  # plus the same number.
  expect_error(gpu_sum(matrix(c(1+2i, 0+3i), 1, 2)), "no complex-valued operation")
})

test_that("complex refusal covers the vector and non-reduction paths too", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  # A bare vector goes through the Exporter's vector_to_mat() branch, and
  # gpu_transpose() reaches the same conversion without reducing anything, so
  # neither can quietly keep working if the refusal is ever narrowed to one
  # entry point.
  expect_error(gpu_sum(c(1+2i, 2+0i)), "complex input")
  expect_error(gpu_transpose(matrix(c(1+2i, 0+3i), 1, 2)), "complex input")
})

test_that("Re() is accepted, so the refusal is about the imaginary part only", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "reduction unavailable on this device")

  # The error tells the caller to do this; it had better work.
  expect_equal(gpu_sum(Re(matrix(c(1+2i, 0+3i), 1, 2))), 1, tolerance = tol_f32_reduction)
})
