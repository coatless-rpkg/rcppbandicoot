# The 4 GiB device-buffer refusal (Rcpp::traits::bandicoot_check_alloc, in
# inst/include/RcppBandicootAs.h).
#
# Once a buffer reaches 2^32 bytes the drivers measured here truncate the
# request modulo 2^32, return CL_SUCCESS, and hand back a correctly shaped
# matrix full of whatever was never written: gpu_randu(32768, 32768) came back
# all zeros, gpu_randu(33000, 33000) filled 1.4011% of its entries where the
# truncation predicts 0.014011%, and gpu_sum(matrix(1, 2^30, 1)) returned 0.
# Nothing raised. The guard turns all of those into R errors.
#
# Nothing in this file reaches the device, and that is the point rather than an
# accident: a guard that fires after the truncated buffer exists is no guard.
# It is also what lets the file live in the core suite instead of needing its
# own process -- see tests/testthat.R -- and what makes it safe for it to sort
# ahead of test-probe.R. Touching the device before the first
# cached_gpu_device_info() call poisons the OpenCL context for the rest of the
# session, for the upstream reason helper-gpu.R sets out at length; the one case
# that cannot avoid a device (a product larger than both its operands, which is
# only knowable after both have been converted) is in
# test-gpu-op-matrix-multiply.R instead.
#
# Every test here is opt-in via RCPPBANDICOOT_RUN_LARGE_ALLOC; the reasoning for
# gating cases that cost nothing is on skip_unless_large_alloc() in helper-gpu.R.

# The message has to be specific enough that these cannot be satisfied by some
# unrelated failure -- "no device", "bad_alloc", a segfault caught as an error.
# It is the refusal that is under test, not the absence of a result.
refusal_pattern <- "refuses any single device buffer of 2\\^32 bytes"

test_that("a matrix arriving from R at the limit is refused", {
  skip_unless_large_alloc()

  # 2^30 float elements is exactly 2^32 bytes. Built as an ALTREP compact
  # sequence, which costs nothing until something materialises it; setting dim()
  # keeps it compact, so this line allocates no more than a scalar. The Exporter
  # reads the dim attribute and refuses before it touches INTEGER(), which is
  # what would turn this into 4 GiB.
  x <- 1:2^30
  dim(x) <- c(2^30L, 1L)

  expect_error(gpu_sum(x), refusal_pattern)
})

test_that("a size given as dimensions rather than data is refused", {
  skip_unless_large_alloc()

  # Neither of these has an R object to be caught by on the way in: the size
  # comes from a pair of small integers, so the Exporters never see it and the
  # check has to live in the exported function itself.
  expect_error(gpu_randu(32768L, 32768L), refusal_pattern)   # 2^30 elements
  expect_error(gpu_eye(65536L), refusal_pattern)             # 2^32 elements
})

test_that("the refusal reports the boundary it actually applies", {
  skip_unless_large_alloc()

  # A guard that refused everything would satisfy every test above and be
  # useless, and one placed an element or a power of two out would satisfy them
  # too. The natural way to pin the boundary is to allocate just under it and
  # watch it succeed -- but "just under" is 4 GiB of real device memory, and a
  # device without it would fail for exactly the unrelated reason this guard was
  # written not to pre-empt. So the boundary is checked through the arithmetic
  # the message reports instead, which costs nothing and says the same thing.
  #
  # 32768^2 = 1073741824 elements, 4 bytes each, 4294967296 bytes = 2^32
  # exactly: the smallest request that is refused. One element fewer is
  # permitted, so the stated ceiling is 1073741823.
  msg <- tryCatch(gpu_randu(32768L, 32768L), error = conditionMessage)

  expect_match(msg, "1073741824 elements at 4 bytes each is 4294967296 bytes",
               fixed = TRUE)
  expect_match(msg, "largest permitted for this element type is 1073741823 elements",
               fixed = TRUE)
})
