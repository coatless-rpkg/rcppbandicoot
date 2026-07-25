test_that("bandicoot_version() returns the vendored Bandicoot version", {
  res <- bandicoot_version()
  expect_type(res, "character")
  expect_length(res, 1L)
  # Bare "major.minor.patch" with no release name, so that downstream code can
  # feature-gate with package_version() -- coot_version::as_string() appends
  # "(Bandwidth Glutton)", which package_version() rejects.
  expect_match(res, "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  expect_s3_class(package_version(res), "package_version")
  # inst/version.txt is what the upstream-update workflow bumps; the compiled
  # constant and the file must not drift apart.
  expect_identical(
    res,
    readLines(system.file("version.txt", package = "RcppBandicoot"))[1]
  )
})

test_that("bandicoot_config() reports a usable build-time configuration", {
  out <- capture.output(cfg <- bandicoot_config())

  expect_type(cfg, "list")
  expect_true(all(c("opencl", "cuda", "clblast", "clblas", "openmp",
                    "default_backend") %in% names(cfg)))
  for (nm in c("opencl", "cuda", "clblast", "clblas", "openmp")) {
    expect_type(cfg[[nm]], "logical")
    expect_length(cfg[[nm]], 1L)
  }

  # configure aborts the build outright when neither backend is present, so a
  # package that loaded at all must report one.  Values are never asserted:
  # they are configure-time constants and machine-specific.
  expect_true(cfg$opencl || cfg$cuda)
  expect_true(cfg$default_backend %in% c("CL_BACKEND", "CUDA_BACKEND"))
  expect_true(any(grepl("RcppBandicoot Configuration", out, fixed = TRUE)))
})

test_that("bandicoot_config(verbose = TRUE) adds the flag strings", {
  invisible(capture.output(cfg <- bandicoot_config(verbose = TRUE)))
  expect_type(cfg$cxxflags, "character")
  expect_type(cfg$libs, "character")
})
