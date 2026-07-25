# Device RNG seeding.
#
# Bandicoot defaults its device RNG seed to 0 and never randomises it, so
# gpu_randu() used to return bit-identical values in every fresh R process --
# an N-process Monte Carlo run yielded zero independent information, silently.
# gpu_randu() now seeds once per session from R's own stream, and
# gpu_set_seed() overrides that explicitly for reproducibility.

test_that("gpu_set_seed() rejects values that cannot be a seed", {
  # Device-free: validation happens before the runtime is touched.
  expect_error(gpu_set_seed(-1), "non-negative")
  expect_error(gpu_set_seed(1.5), "whole number")
  expect_error(gpu_set_seed(NA_real_), "whole number")
  expect_error(gpu_set_seed(Inf), "finite")
})

test_that("gpu_set_seed() makes gpu_randu() reproducible within a session", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_randu", "device RNG unavailable on this device")

  gpu_set_seed(42)
  first <- gpu_randu(3, 2)
  gpu_set_seed(42)
  second <- gpu_randu(3, 2)

  expect_identical(dim(first), c(3L, 2L))
  expect_identical(first, second)
})

test_that("different seeds produce different draws", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_randu", "device RNG unavailable on this device")

  gpu_set_seed(1)
  a <- gpu_randu(4, 4)
  gpu_set_seed(2)
  b <- gpu_randu(4, 4)

  expect_false(isTRUE(all.equal(a, b)))
})

test_that("a fresh session does not reproduce the previous session's draw", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_randu", "device RNG unavailable on this device")
  skip_on_cran()

  # The actual regression being guarded is cross-process, so it needs a
  # subprocess: within one session the lazy seed is drawn only once.
  draw_in_subprocess <- function() {
    out <- suppressWarnings(system2(
      file.path(R.home("bin"), "Rscript"),
      c("-e", shQuote(
        'library(RcppBandicoot); cat(sprintf("%.12f", gpu_randu(2, 2)[1, 1]))'
      )),
      stdout = TRUE, stderr = FALSE
    ))
    utils::tail(out[nzchar(out)], 1L)
  }

  first <- draw_in_subprocess()
  second <- draw_in_subprocess()

  # tail() of an empty vector is character(0), and identical(character(0),
  # character(0)) is TRUE -- so without the length guard the intended skip
  # falls through and reports a spurious failure instead.
  skip_if(length(first) != 1L || length(second) != 1L ||
            !nzchar(first) || !nzchar(second),
          "subprocess did not produce a draw (no device in the child session)")
  expect_false(identical(first, second))
})
