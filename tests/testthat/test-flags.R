test_that("RcppBandicootCxxFlags() points at the installed headers and kernels", {
  flags <- RcppBandicootCxxFlags()
  inc <- system.file("include", package = "RcppBandicoot")

  expect_match(flags, inc, fixed = TRUE)
  expect_match(flags, "-DCOOT_DEFAULT_BACKEND=")

  # Regression guard for the downstream-consumer bug: without this token,
  # sourceCpp'd code falls back to a "bandicoot_bits/kernels/" path that this
  # package does not ship.
  expect_match(flags, "-DCOOT_KERNEL_SOURCE_DIR=", fixed = TRUE)
  ks <- sub(".*COOT_KERNEL_SOURCE_DIR='\"([^\"]*)\".*", "\\1", flags)
  expect_true(dir.exists(file.path(ks, "opencl", "defs")))
  expect_true(file.exists(file.path(ks, "opencl", "defs", "opencl_prelims.cl")))
})

test_that("CxxFlags() and LdFlags() print and return their counterparts", {
  expect_output(cxx <- CxxFlags())
  expect_identical(cxx, RcppBandicootCxxFlags())
  expect_output(ld <- LdFlags())
  expect_identical(ld, RcppBandicootLdFlags())
  expect_true(nzchar(RcppBandicootLdFlags()))
})

test_that("inlineCxxPlugin() carries the package flags and no defunct standard", {
  plugin <- inlineCxxPlugin()
  expect_type(plugin, "list")
  expect_true(startsWith(plugin$env$PKG_CXXFLAGS, RcppBandicootCxxFlags()))
  # `R CMD config CXX14` is defunct as of R 4.5; setting USE_CXX14 breaks
  # sourceCpp() on every supported R version.
  expect_null(plugin$env$USE_CXX14)
})
