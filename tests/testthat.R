# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

library(testthat)
library(RcppBandicoot)

# Core suite only.  A failing GPU operation is an access violation
# (SIGSEGV / 0xC0000005), not an R condition, so it kills the R process and
# every result after it.  Each test-gpu-op-*.R therefore gets its own top-level
# script (tests/zz-gpu-<op>.R), which R CMD check runs as a separate R CMD BATCH
# process; the filter below keeps them out of this one so a crash in any single
# operation cannot destroy the device-free results.
test_check("RcppBandicoot", filter = "^(?!gpu-op-)", perl = TRUE)
