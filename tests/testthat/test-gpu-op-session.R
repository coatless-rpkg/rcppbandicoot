# Several different operations back-to-back in ONE session.
#
# Every other file in this directory runs exactly one operation in a fresh
# process, which is the most favourable configuration a JIT codegen fault can
# be given: nothing else has been compiled, nothing else is resident, and the
# runtime is as close to its initial state as it ever gets.  Per-operation
# results are therefore a LOWER bound on the failure rate a real user sees --
# real code calls eye, then transpose, then a reduction, in the same process.
# This file is the only coverage of that, so it is deliberately the one place
# where a crash can take several operations down with it.
#
# It has its own xfail key ("gpu_session") rather than reusing the per-op keys:
# a device where each operation is individually fine but the combination is not
# must stay distinguishable from a device where a single operation is broken.

test_that("several operations run back-to-back in one session", {
  skip_if_no_gpu()
  skip_if_xfail("gpu_session",
                "multi-operation session unavailable on this device")

  # Ordered so that each step feeds the next: a fault that only appears once a
  # previous kernel is resident shows up here and nowhere else.
  I3 <- gpu_eye(3)
  expect_equal(I3, diag(3), tolerance = tol_f32_elementwise)

  A <- matrix(as.numeric(1:6), nrow = 2, ncol = 3)
  At <- gpu_transpose(A)
  expect_equal(At, t(A), tolerance = tol_f32_elementwise)

  expect_equal(gpu_sum(At), sum(A), tolerance = tol_f32_reduction)
  expect_equal(gpu_mean(At), mean(A), tolerance = tol_f32_reduction)
})
