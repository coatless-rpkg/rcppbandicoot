test_that("the device probes never throw and agree with one another", {
  # gpu_available()/gpu_device_info()/gpu_initialize() each call
  # coot::coot_init() directly and are not safe to call more than once,
  # combined, in the same process -- see the long comment above
  # cached_gpu_device_info() in helper-gpu.R for the reproduced upstream
  # cause. By the time this file runs, the gpu-*.R suites have already made
  # the session's one safe call (through that cache), so this test reads the
  # same cached gpu_device_info() rather than issuing a fresh raw
  # gpu_available() + gpu_device_info() pair, which reliably disagree with
  # each other once other kernels are already warm.
  info <- cached_gpu_device_info()
  # Pin the field the C++ layer actually returns, before coercion: asserting
  # against isTRUE(info$available) would be vacuous, since isTRUE always yields
  # a scalar non-NA logical no matter what the C++ produced.
  expect_type(info$available, "logical")
  expect_length(info$available, 1L)
  expect_false(is.na(info$available))
  available <- isTRUE(info$available)

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
  # Compared against the cached probe, not a fresh gpu_available() call, for
  # the same reason as above: gpu_initialize() here is itself a second
  # explicit init in the session, and a further gpu_available() call right
  # after it is exactly the sequence that reproduces the upstream defect.
  expect_identical(res, isTRUE(cached_gpu_device_info()$available))
})

test_that("a GPU call raises an R condition rather than crashing with no device", {
  if (isTRUE(cached_gpu_device_info()$available)) {
    skip("a device is present; the no-device error path is not reachable here")
  }
  # Bandicoot throws std::runtime_error, which Rcpp's END_RCPP converts into a
  # normal R error.  Anything else (a segfault, an abort) would take the whole
  # session down and is the single worst outcome on a CRAN flavour.
  expect_error(gpu_eye(2))
})
