# Layout drift guard for the one-process-per-operation split.
#
# Each test-gpu-op-<op>.R is only ever executed because tests/zz-gpu-<op>.R
# exists to run it: tests/testthat.R filters the per-operation files out of the
# core suite, so an op file with no runner is silently dead code -- it looks
# like coverage in the repository and executes nowhere.  The reverse (a runner
# whose op file was renamed or removed) makes test_check() match nothing, which
# testthat reports as an error rather than a pass, but is still worth naming
# here so the two lists are checked in one place.

test_that("every per-operation test file has a matching runner script", {
  # Under test_check()/test_dir()/devtools::test() the working directory is
  # tests/testthat, so the runners are one level up.  Under pkgload, or when a
  # single file is sourced by hand, there is no tests/ layout to inspect and
  # this check has nothing to say.
  testthat_dir <- getwd()
  tests_dir <- dirname(testthat_dir)
  skip_if_not(
    basename(testthat_dir) == "testthat" &&
      file.exists(file.path(tests_dir, "testthat.R")),
    "not running from a tests/ directory layout"
  )

  ops_with_tests <- sort(sub(
    "\\.R$", "",
    sub("^test-gpu-op-", "",
        list.files(testthat_dir, pattern = "^test-gpu-op-.+\\.R$"))
  ))
  ops_with_runners <- sort(sub(
    "\\.R$", "",
    sub("^zz-gpu-", "",
        list.files(tests_dir, pattern = "^zz-gpu-.+\\.R$"))
  ))

  # A silent zero-on-both-sides would make the comparison below vacuous, which
  # is precisely the regression (every op file gone, or the naming scheme
  # changed) this file exists to catch.
  expect_gt(length(ops_with_tests), 0L)

  # Named variables rather than expect_setequal() so the failure output says
  # which operation is unpaired and in which direction.
  ops_missing_a_runner <- setdiff(ops_with_tests, ops_with_runners)
  expect_equal(ops_missing_a_runner, character())

  runners_missing_a_test <- setdiff(ops_with_runners, ops_with_tests)
  expect_equal(runners_missing_a_test, character())
})
