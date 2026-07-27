# Compiler warnings under GCC 14 with bandicoot 4.0.0

## Environment

- Bandicoot 4.0.0 ("Bandwidth Glutton")
- GCC 14.3.0 on Windows (MinGW / Rtools44); reproducible on Linux/glibc with the same compiler
- Surfaced while building [RcppBandicoot](https://github.com/coatless-rpkg/rcppbandicoot), which embeds the bandicoot headers verbatim

## Overview

Building bandicoot 4.0.0 under GCC 14 emits **1,682 compiler warnings** spanning four distinct diagnostic classes. None affect runtime correctness — the generated code is fine, the warnings are all compile-only — but collectively they trip `R CMD check` at the WARNING status, which blocks CRAN release for any R package embedding the bandicoot headers (RcppBandicoot, and any future downstream binding).

The warnings cluster in three files:

- **`opencl/runtime_bones.hpp` + `runtime_meat.hpp`** — two related issues in the `runtime_t::adapt_uword` class. The primary constructor writes only one of `val32` / `val64` depending on a runtime branch (`sizeof(uword) >= 8 && has_sizet64()`), but the copy and move constructors read both unconditionally, so the unwritten branch trips `-Wmaybe-uninitialized`. Separately, those same copy/move constructors list their mem-initializers in an order that doesn't match the struct declaration (`val32` before `val64` in the init list, `val64` before `val32` in the declaration), triggering `-Wreorder`.

- **`kernel_gen/array_util.hpp:78`** — a SFINAE check of the form `decltype(T::len, void())`. `T::len` resolves to a static member function, so in this comma expression its name decays to a function pointer that is then immediately discarded. GCC 14's tightened `-Waddress` treats that as "did you forget to call this?" and fires — 1,534 times across template instantiations.

- **`mtglue_mixed_meat.hpp:34-35`** — two `typedef`s (`in_eT1`, `in_eT2`) at the top of `mtglue_mixed_times::apply` that are never referenced in the function body.

Each has a small local fix, all preserving existing semantics. Patches are below.

## Warning tally

| Warning | Count | File |
|---|---|---|
| `-Waddress` | 1534 | `kernel_gen/array_util.hpp:78` |
| `-Wmaybe-uninitialized` | 116 | `opencl/runtime_meat.hpp:1723,1724` |
| `-Wreorder` | 24 (8 unique × 3 TUs) | `opencl/runtime_meat.hpp:1703,1721` |
| `-Wunused-local-typedefs` | 8 | `mtglue_mixed_meat.hpp:34,35` |

---

## 1. `-Wreorder` in `opencl/runtime_meat.hpp`

Declaration order in `runtime_bones.hpp:274-277` is `size, addr, val64, val32`. The copy- and move-constructor mem-initializer lists list `val32` before `val64`, so GCC reorders and warns.

```diff
--- a/include/bandicoot_bits/opencl/runtime_meat.hpp
+++ b/include/bandicoot_bits/opencl/runtime_meat.hpp
@@ -1702,8 +1702,8 @@ runtime_t::adapt_uword::adapt_uword(const uword val)
 inline
 runtime_t::adapt_uword::adapt_uword(const runtime_t::adapt_uword& other)
   : size(other.size)
-  , val32(other.val32)
   , val64(other.val64)
+  , val32(other.val32)
   {
   if (other.addr == &other.val32)
     {
@@ -1720,8 +1720,8 @@ runtime_t::adapt_uword::adapt_uword(const runtime_t::adapt_uword& other)
 inline
 runtime_t::adapt_uword::adapt_uword(runtime_t::adapt_uword&& other)
   : size(other.size)
-  , val32(other.val32)
   , val64(other.val64)
+  , val32(other.val32)
   {
   if (other.addr == &other.val32)
     {
```

---

## 2. `-Wmaybe-uninitialized` in `opencl/runtime_meat.hpp`

The primary constructor at `runtime_meat.hpp:1682` writes only **one** of `val32` / `val64` depending on the runtime branch (`sizeof(uword)` and `has_sizet64()`). The copy/move constructors then unconditionally read both, so the branch that wasn't written is formally uninitialized — the warning is correct. Cheapest fix is default-initializing both in the class:

```diff
--- a/include/bandicoot_bits/opencl/runtime_bones.hpp
+++ b/include/bandicoot_bits/opencl/runtime_bones.hpp
@@ -271,10 +271,10 @@ class runtime_t::adapt_uword
   {
   public:
 
-  coot_aligned size_t size;
-  coot_aligned void*  addr;
-  coot_aligned u64    val64;
-  coot_aligned u32    val32;
+  coot_aligned size_t size  = 0;
+  coot_aligned void*  addr  = nullptr;
+  coot_aligned u64    val64 = 0;
+  coot_aligned u32    val32 = 0;
 
   inline adapt_uword(const uword val = 0); // default value needed for allocating several at once
```

(Also initializing `size` / `addr` for symmetry, so a caller ever reading them before the body assigns wouldn't trip `-Wuninitialized` either.)

---

## 3. `-Waddress` in `kernel_gen/array_util.hpp`

```cpp
template<typename T>
struct has_len_member<T, decltype(T::len, void())>
  { static const bool value = true; };
```

`T::len` resolves to a static member function (see the `static inline constexpr size_t len()` definitions throughout `array_util.hpp`). In the comma-operator form, the unparenthesized name decays to a function pointer that is then immediately discarded — GCC 14's tightened `-Waddress` treats this as "you took the address of a function and threw it away, did you mean to call it?" Casting to `void` explicitly signals the discard and silences the warning, without changing SFINAE behavior (both forms yield `void` iff `T::len` is a well-formed name and trigger substitution failure otherwise):

```diff
--- a/include/bandicoot_bits/kernel_gen/array_util.hpp
+++ b/include/bandicoot_bits/kernel_gen/array_util.hpp
@@ -75,7 +75,7 @@ struct has_len_member
   };
 
 template<typename T>
-struct has_len_member<T, decltype(T::len, void())>
+struct has_len_member<T, decltype((void)T::len)>
   {
   static const bool value = true;
   };
```

This one-line change kills all 1534 instantiations of the warning.

---

## 4. `-Wunused-local-typedefs` in `mtglue_mixed_meat.hpp`

`in_eT1` / `in_eT2` are declared at the top of `mtglue_mixed_times::apply` but never used inside the function body — the later `PT1` / `PT2` typedefs are derived from `partial_unwrap<...>` directly.

```diff
--- a/include/bandicoot_bits/mtglue_mixed_meat.hpp
+++ b/include/bandicoot_bits/mtglue_mixed_meat.hpp
@@ -31,9 +31,6 @@ mtglue_mixed_times::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_
   {
   coot_debug_sigprint();
 
-  typedef typename T1::elem_type in_eT1;
-  typedef typename T2::elem_type in_eT2;
-
   // For mixed matrix multiplication, we have to convert both results to the output type.
   const partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>> tmp1(mtOp<out_eT, T1, mtop_conv_to>(X.A));
   const partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>> tmp2(mtOp<out_eT, T2, mtop_conv_to>(X.B));
```

---

## Downstream context

RcppBandicoot has applied the four patches above to its embedded header copies and verified they silence every warning category on GCC 14. Adopting them upstream would let R packages carrying bandicoot drop any workaround flags and pass `R CMD check` clean on CRAN's Windows and Linux builders.

Happy to open MRs against `unstable` if preferred.

---
---

# `accu()` returns a wrong value on the multi-pass reduction path

Second, unrelated report. Draft — **not filed**.

## Overview

`accu()` on a matrix large enough to leave the single-pass reduction returns a
wrong value on two independent OpenCL implementations, and a *different* wrong
value on each, while a third gets it right. Nothing errors: the call succeeds
and hands back a plausible number.

## Observed

Input is a 400 x 400 single-precision matrix of `runif` values (seed 7), whose
sum is 79993.0. 400 x 400 = 1.6e5 elements, which crosses out of the single-pass
`_small` kernel — the boundary sits near 1.5e5 with `CL_KERNEL_WORK_GROUP_SIZE`
4096.

| Platform | OpenCL implementation | Compiler | Result |
|---|---|---|---|
| Linux x86-64 | PoCL 5.0+debian | LLVM 16.0.6 | 79993.0 — correct |
| Windows x86-64 | Intel oclcpuexp 2025-WW13 | clang 20.0.0git | **25040.2** |
| macOS arm64 | PoCL 7.1 (conda-forge) | LLVM 19.1.7 | **NaN** |

All three enumerate a healthy fp64 CPU device and compute every other tested
operation correctly — transpose, matrix add, matrix multiply, elementwise
square, mean, eye, randu, and the single-pass `accu()` — and each passes a
kernel round trip that builds, launches, reads back and checks the numbers
before any of this runs.

25040.2 is roughly a third of the correct total, which would be consistent with
a partial reduction being returned as the whole. That is inference, not
measurement, and it does not account for the NaN on macOS.

## Why this reads as the reduction rather than the runtimes

Two implementations from different vendors, on different LLVM generations (20
and 19) and different CPU architectures (x86-64 and arm64), disagreeing with the
correct answer in two different ways, while a third implementation on the same
source is correct. A miscompile would have to reproduce across both toolchains
and produce distinct wrong answers.

For context on how these platforms were reached: both were previously crashing
outright on nearly every operation, for causes that did turn out to be
toolchain-specific and are now excluded — Intel's LLVM 22 build of oclcpuexp
(access violation in `common_clang64.dll` on every kernel) and Homebrew's
llvm@21 PoCL (llvm#161077, kernel-stub codegen). Pinning away from both left
exactly this one failure standing on each.

## Reproducing

Any `accu()` over enough elements to leave the single-pass path. Through the R
bindings, where `gpu_sum()` is a thin wrapper over `coot::accu()` on a
`coot::fmat`:

```r
set.seed(7)
A <- matrix(runif(400 * 400), 400, 400)
gpu_sum(A) - sum(A)   # 0 on Linux/PoCL 5.0; ~-5.5e4 on Windows; NaN on macOS
```

## Not established

- No minimal C++ reproducer outside the R bindings.
- RcppBandicoot's own comment at the call site suspects a device mis-reporting
  its subgroup size, letting the multi-pass kernel read past the end of
  `__local`. That is a hypothesis inherited from an earlier diagnosis; nothing
  here confirms it and the kernel has not been instrumented.
- Only CPU implementations tested; no real GPU.
- Whether the Windows and macOS symptoms share one cause or are two.

## A second operation shows the same proportion

`mean()` over a wide matrix fails the same way on the same runtime, which is
why this reads as one defect rather than two.

A 10 x 1e4 matrix of `runif` values, seed 23, whose true mean is 0.500170:

| Platform | OpenCL implementation | Result |
|---|---|---|
| Linux x86-64 | PoCL 5.0+debian / LLVM 16 | correct |
| macOS arm64 | PoCL 7.1 / LLVM 19.1.7 | correct |
| Apple (local, arm64 GPU) | Apple OpenCL | 0.500170 -- exact |
| Windows x86-64 | Intel oclcpuexp 2025-WW13 / clang 20 | **0.16** |

0.16 / 0.50 = 0.32, against 25040.2 / 79993.0 = 0.313 for the `accu()` case
above. Two different reductions, two different entry points, the same runtime,
and both land near a third of the correct value -- which is what only some of
the work-groups contributing would look like. The narrow case of the same
operation (`1e5 x 2`) is correct on that runtime, so it tracks the shape of the
reduction rather than the operation.

Reproducing, alongside the `accu()` case:

```r
set.seed(23)
w <- matrix(runif(1e5), 10, 1e4)
gpu_mean(w)   # 0.500170 on PoCL and Apple; about 0.16 on Intel's runtime
```

## Severity

This one is silent. Every other failure these platforms produced was a hard
crash, which announces itself; this returns a number the caller will use.
