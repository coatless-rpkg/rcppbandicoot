# One operation per file, run by tests/zz-gpu-sum-multipass.R in its own R
# process.  A GPU fault here is an access violation, not an R condition, so it
# would take the whole process down; isolation keeps it from destroying every
# other result.

test_that("gpu_sum() is correct on the multi-pass reduction path", {
  skip_if_no_gpu()
  # Excluded independently of the single-pass gpu_sum test in
  # test-gpu-op-sum.R: a device
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
