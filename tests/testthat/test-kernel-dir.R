# The runtime kernel-source directory.
#
# COOT_KERNEL_SOURCE_DIR is compiled as a call to rcppbandicoot_kernel_dir()
# (src/kernel_dir.cpp) and overridden by .onLoad from system.file(), so that a
# binary package works wherever it was installed rather than where it was built.
#
# Nothing used to read that compiled state: test-flags.R inspects the R-side
# string from RcppBandicootCxxFlags(), which is computed independently. So the
# failure mode of a .onLoad regression was that every GPU test SKIPPED -- no
# device, because no kernel could be compiled -- and the suite stayed green.
# These tests are device-free and read the real compiled value.

test_that("the compiled kernel directory points at the installed kernels", {
  dir <- RcppBandicoot:::get_kernel_source_dir()

  expect_type(dir, "character")
  expect_length(dir, 1L)
  expect_true(nzchar(dir))
  expect_true(dir.exists(dir))

  # Bandicoot concatenates "opencl/<file>" onto this string directly, so a
  # missing trailing separator silently yields ".../ksopencl/..." and every
  # kernel load fails with "Cannot open required kernel source."
  expect_match(dir, "/$")

  # The directory has to actually contain the kernel sources, not merely exist.
  expect_true(dir.exists(file.path(dir, "opencl", "defs")))
  expect_true(file.exists(file.path(dir, "opencl", "defs", "opencl_prelims.cl")))
})

test_that("the compiled kernel directory matches the installed location", {
  # This is the property .onLoad exists to establish: the path tracks where the
  # package actually landed, not where it was configured.
  expected <- paste0(
    system.file("include", "bandicoot_bits", "ks", package = "RcppBandicoot"),
    "/"
  )
  expect_identical(RcppBandicoot:::get_kernel_source_dir(), expected)
})
