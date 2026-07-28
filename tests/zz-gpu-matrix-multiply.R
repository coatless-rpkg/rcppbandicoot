# Own script so a GPU access violation in this operation cannot kill the other
# tests: R CMD check runs each tests/*.R file in a separate R process.
library(testthat)
library(RcppBandicoot)

test_check("RcppBandicoot", filter = "^gpu-op-matrix-multiply$")
