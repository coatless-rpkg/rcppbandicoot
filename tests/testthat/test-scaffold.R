# testthat 3.3.2's test_check()/test_dir() calls cli::cli_abort("No test
# files found.") when zero files matching "^test.*\\.[rR]$" exist under
# tests/testthat/ -- an R CMD check ERROR, not a clean 0-test pass.  Until
# Tasks 6-8 add the real suite (test-config.R, test-flags.R, test-probe.R,
# test-gpu-*.R), this file keeps the scaffold from tripping that guard, and
# doubles as a device-free smoke test that the package and its namespace
# exports actually load.
test_that("RcppBandicoot loads and exports the expected symbols", {
  expect_true("package:RcppBandicoot" %in% search())
  expect_true(is.function(gpu_available))
  expect_true(is.function(gpu_initialize))
  expect_true(is.function(gpu_device_info))
  expect_true(is.function(bandicoot_config))
  expect_true(is.function(bandicoot_version))
})
