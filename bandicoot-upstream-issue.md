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
