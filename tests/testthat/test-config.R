test_that("bandicoot_version() prints the vendored Bandicoot version", {
  out <- capture.output(res <- bandicoot_version())
  expect_null(res)
  expect_match(out[1], "^[0-9]+\\.[0-9]+\\.[0-9]+")
  # inst/version.txt is what the upstream-update workflow bumps; the compiled
  # constant and the file must not drift apart.
  expect_identical(
    sub(" .*$", "", out[1]),
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
