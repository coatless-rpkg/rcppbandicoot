test_that("gpu_sum() sums all elements", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_sum", "")

  A <- matrix(as.numeric(1:6), 2, 3)
  expect_equal(gpu_sum(A), 21, tolerance = tol_f32_reduction)
  # R vectors are converted to n x 1 matrices by the Exporter.
  expect_equal(gpu_sum(c(1, 2, 3)), 6, tolerance = tol_f32_reduction)
})

test_that("gpu_sum() is correct on the multi-pass reduction path", {
  skip_if_no_gpu()
  # Excluded independently of the single-pass gpu_sum test above: a device
  # that mis-reports its subgroup size only breaks the multi-pass kernel
  # (see the comment below), so xfail'ing plain "gpu_sum" would also hide
  # the small-input path, which is unaffected and should keep running.
  skip_if_xfail("gpu_sum_multipass",
                "device mis-reports subgroup size; multi-pass reduction reads past __local")

  # Below roughly 1.5e5 elements (with CL_KERNEL_WORK_GROUP_SIZE = 4096) the
  # reduction stays on the single-pass "_small" kernel.  400 x 400 = 1.6e5
  # crosses into the multi-pass kernel, which is where a device that
  # mis-reports its subgroup size reads past the end of __local memory and
  # returns a plausible-looking wrong answer instead of an error.
  set.seed(7)
  A <- matrix(runif(400 * 400), 400, 400)
  expect_equal(gpu_sum(A), sum(A), tolerance = tol_f32_reduction_large)
})

test_that("gpu_mean() computes the grand mean", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_mean", "")

  # mean(mean(A)) is mean-of-column-means, which is the grand mean for any
  # rectangular matrix, and additionally exercises the Row<float> intermediate.
  A <- matrix(as.numeric(1:6), 2, 3)
  expect_equal(gpu_mean(A), 3.5, tolerance = tol_f32_reduction)
  expect_equal(gpu_mean(c(1, 2, 3)), 2, tolerance = tol_f32_reduction)

  set.seed(19)
  B <- matrix(rnorm(40 * 25), 40, 25)
  expect_equal(gpu_mean(B), mean(B), tolerance = tol_f32_reduction)
})
