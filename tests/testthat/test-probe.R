test_that("the device probes never throw and agree with one another", {
  available <- gpu_available()
  expect_type(available, "logical")
  expect_length(available, 1L)
  expect_false(is.na(available))

  info <- gpu_device_info()
  expect_type(info, "list")
  expect_true(all(c("available", "backend", "fp64", "fp16", "subgroups",
                    "subgroup_size", "n_units", "max_wg") %in% names(info)))
  expect_true(info$backend %in% c("none", "opencl", "cuda"))

  if (available) {
    expect_true(info$available)
    expect_gt(info$n_units, 0)
  } else {
    expect_false(info$available)
    expect_identical(info$backend, "none")
    # Pin the full no-device contract, not just two keys: the remaining six are
    # hardcoded C++ defaults on the early-return path, and this is the branch
    # CRAN and every device-free run actually take. expect_identical also pins
    # the type (logical vs the doubles), so a field silently becoming NA or an
    # integer would be caught here.
    expect_identical(info[c("fp64", "fp16", "subgroups")],
                     list(fp64 = FALSE, fp16 = FALSE, subgroups = FALSE))
    expect_identical(info[c("subgroup_size", "n_units", "max_wg")],
                     list(subgroup_size = 0, n_units = 0, max_wg = 0))
  }
})

test_that("gpu_initialize() returns a scalar logical", {
  res <- gpu_initialize()
  expect_type(res, "logical")
  expect_length(res, 1L)
  expect_identical(res, gpu_available())
})

test_that("a GPU call raises an R condition rather than crashing with no device", {
  if (gpu_available()) {
    skip("a device is present; the no-device error path is not reachable here")
  }
  # Bandicoot throws std::runtime_error, which Rcpp's END_RCPP converts into a
  # normal R error.  Anything else (a segfault, an abort) would take the whole
  # session down and is the single worst outcome on a CRAN flavour.
  expect_error(gpu_eye(2))
})
