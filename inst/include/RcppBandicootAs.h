// RcppBandicootAs.h: As functions for Rcpp/Bandicoot glue
//
// Copyright (C) 2023-2025 James Balamuta
//
// This file is part of RcppBandicoot.
//
// RcppBandicoot is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RcppBandicoot is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RcppBandicoot.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RcppBandicoot__RcppBandicootAs__h
#define RcppBandicoot__RcppBandicootAs__h

#include <limits>
#include <type_traits>

namespace Rcpp {
    namespace traits {

        // Whether the destination element type can hold an imaginary part.
        template <typename T>
        struct bandicoot_is_complex {
            static const bool value = false;
        };

        template <typename U>
        struct bandicoot_is_complex< std::complex<U> > {
            static const bool value = true;
        };

        // Helper function to convert from Rcomplex to complex type.  Defined
        // only for the complex element types, so there is no instantiation
        // that could quietly return something else.
        template <typename T>
        struct complex_converter;

        // Specialization for std::complex<double>
        template <>
        struct complex_converter<std::complex<double>> {
            static inline std::complex<double> from_rcomplex(const Rcomplex& src) {
                return std::complex<double>(src.r, src.i);
            }
        };

        // Specialization for std::complex<float>
        template <>
        struct complex_converter<std::complex<float>> {
            static inline std::complex<float> from_rcomplex(const Rcomplex& src) {
                return std::complex<float>(static_cast<float>(src.r), static_cast<float>(src.i));
            }
        };

        // Copying a CPLXSXP payload, split on whether the destination element
        // type has room for the imaginary part.  The split is what makes the
        // refusal below unavoidable: the "no room" case has no conversion to
        // perform at all, so there is no code anywhere in this header that
        // keeps the real part and drops the rest.
        //
        // Dropping it is exactly what used to happen. A one-line
        // static_cast<T>(src.r) made gpu_sum(matrix(c(1+2i, 0+3i), 1, 2))
        // return 1 where base R's sum() returns 1+5i -- a plausible number, no
        // warning, wrong. That is the same silent-substitution failure the
        // NA handling below was added to close, arriving through the same
        // function by a different branch.
        //
        // The package exports no complex-valued operation, so there is nothing
        // to route complex input to and no complex code path worth inventing
        // here: one would only carry the values further before they were
        // dropped somewhere less visible. Refusing at the boundary is the
        // whole of the correct behaviour.
        template <typename T, bool dest_is_complex = bandicoot_is_complex<T>::value>
        struct bandicoot_complex_copy {
            static inline void run(const Rcomplex*, T*, coot::uword, const char* target) {
                Rcpp::stop("Cannot convert complex input to a real-valued %s: "
                           "the imaginary part has no representation there, and "
                           "RcppBandicoot exports no complex-valued operation. "
                           "Pass Re(x) if discarding the imaginary part is what "
                           "was meant.", target);
            }
        };

        template <typename T>
        struct bandicoot_complex_copy<T, true> {
            static inline void run(const Rcomplex* src, T* dest, coot::uword n_elem, const char*) {
                for (coot::uword i = 0; i < n_elem; ++i) {
                    dest[i] = complex_converter<T>::from_rcomplex(src[i]);
                }
            }
        };

        // Whether the destination element type has room for a missing value.
        // A floating-point element has NaN to spare, and so does a complex one
        // built out of floating-point parts. An integral element does not: every
        // bit pattern already means a number.
        template <typename T>
        struct bandicoot_na_traits {
            static const bool has_nan = std::is_floating_point<T>::value;
            static inline T na_value() { return std::numeric_limits<T>::quiet_NaN(); }
        };

        template <typename U>
        struct bandicoot_na_traits< std::complex<U> > {
            static const bool has_nan = std::is_floating_point<U>::value;
            static inline std::complex<U> na_value() {
                const U q = std::numeric_limits<U>::quiet_NaN();
                return std::complex<U>(q, q);
            }
        };

        // Refuse any single device buffer of 2^32 bytes (4 GiB) or more.
        //
        // Bandicoot sizes an OpenCL buffer with
        //   clCreateBuffer(ctxt, CL_MEM_READ_WRITE, sizeof(ceT) * n_elem, ...)
        // (bandicoot_bits/opencl/runtime_meat.hpp, runtime_t::acquire_memory)
        // and guards it with exactly two checks: a conform check against
        // Datum<size_t>::max / sizeof(eT), which is about 4.6e18 elements on a
        // 64-bit host and so never fires, and coot_check_bad_alloc on the
        // status clCreateBuffer returns. At and above 2^32 bytes the drivers
        // measured here truncate the request modulo 2^32, hand back a buffer
        // sized by the remainder, and still report CL_SUCCESS -- so
        // coot_check_bad_alloc has nothing to object to, and neither does
        // anything downstream.
        //
        // What R gets back is a correctly shaped matrix that is mostly whatever
        // was never written. Measured on a real device: gpu_randu(32768, 32768)
        // returned all zeros, gpu_randu(33000, 33000) filled 1.4011% of its
        // entries where the truncation predicts 0.014011%, and
        // gpu_sum(matrix(1, 2^30, 1)) returned 0. None of those raised anything.
        //
        // 2^32 bytes is 2^30 elements of float or 2^29 of double. The cut is
        // deliberately at the wrap and not at the device's real capacity: an
        // allocation below it that the device cannot satisfy still fails
        // honestly through coot_check_bad_alloc, and this guard has no business
        // pre-empting that. It covers only the range where failure is silent.
        //
        // The element count arrives as a double and callers form it as a double
        // product, so the overflow this exists to catch cannot occur on the way
        // to the check itself. Every count reachable here is far below 2^53,
        // where doubles still represent integers exactly.
        template <typename T>
        inline void bandicoot_check_alloc(double n_elem, const char* target) {
            const double max_bytes  = 4294967296.0;  // 2^32
            const double max_n_elem = max_bytes / static_cast<double>(sizeof(T));

            if (n_elem >= max_n_elem) {
                Rcpp::stop("Cannot allocate the requested %s: %.0f elements at %.0f "
                           "bytes each is %.0f bytes. RcppBandicoot refuses any single "
                           "device buffer of 2^32 bytes (4 GiB) or more, because the "
                           "byte count wraps at that point and the driver returns a "
                           "silently truncated buffer instead of an error. The largest "
                           "permitted for this element type is %.0f elements.",
                           target,
                           n_elem,
                           static_cast<double>(sizeof(T)),
                           n_elem * static_cast<double>(sizeof(T)),
                           max_n_elem - 1.0);
            }
        }

        // Copy an R vector's payload into a plain C++ buffer of element type T.
        //
        // R spells a missing integer or logical as INT_MIN, which to C++ is an
        // ordinary int, so a bare static_cast turns NA into -2147483648 and every
        // sum, mean or product computed downstream silently reports that number
        // instead of NA. That is not a corner case: matrix(1:6, 2) is an INTEGER
        // matrix, so it is what the documented example inputs go through.
        //
        // REALSXP needs no test on the way in when the destination is
        // floating-point: NA_real_ is a NaN with a payload, so it propagates
        // through arithmetic on its own. Narrowing it to float keeps it missing
        // -- IEEE 754 turns a double NaN into a float NaN -- but drops the
        // payload, so an NA that passes through an fmat comes back to R as NaN
        // rather than NA. That is inherent to single precision, not something
        // this conversion can preserve.
        //
        // An integral destination has nowhere to put a missing value, so any NA
        // or NaN reaching one is refused instead of being quietly cast (which is
        // also undefined behaviour for NaN).
        template <typename T>
        inline void bandicoot_copy_from_r(SEXP source, T* dest, coot::uword n_elem, const char* target) {
            const int sexp_type = TYPEOF(source);

            if (sexp_type == REALSXP) {
                const double* src = REAL(source);
                for (coot::uword i = 0; i < n_elem; ++i) {
                    if (!bandicoot_na_traits<T>::has_nan && ISNAN(src[i])) {
                        Rcpp::stop("Cannot convert NA/NaN to an integer-typed %s: "
                                   "integer elements have no missing-value representation", target);
                    }
                    dest[i] = static_cast<T>(src[i]);
                }
            } else if (sexp_type == INTSXP || sexp_type == LGLSXP) {
                // NA_LOGICAL and NA_INTEGER are both R_NaInt, so one test covers both
                const int* src = (sexp_type == INTSXP) ? INTEGER(source) : LOGICAL(source);
                for (coot::uword i = 0; i < n_elem; ++i) {
                    if (src[i] == NA_INTEGER) {
                        if (!bandicoot_na_traits<T>::has_nan) {
                            Rcpp::stop("Cannot convert NA to an integer-typed %s: "
                                       "integer elements have no missing-value representation", target);
                        }
                        dest[i] = bandicoot_na_traits<T>::na_value();
                    } else {
                        dest[i] = static_cast<T>(src[i]);
                    }
                }
            } else if (sexp_type == CPLXSXP) {
                // No NA test on this branch: reaching the copy at all means the
                // destination is complex, which means it is built out of
                // floating-point parts, so NA_complex_ arrives as a NaN pair
                // and propagates on its own exactly as REALSXP does above.
                // Every other destination is refused outright.
                bandicoot_complex_copy<T>::run(COMPLEX(source), dest, n_elem, target);
            } else {
                Rcpp::stop("Unsupported SEXP type for conversion to %s", target);
            }
        }

        // Exporter for coot::Mat<T> - convert R matrix to Bandicoot matrix
        template <typename T>
        class Exporter< coot::Mat<T> > {
        public:
            Exporter(SEXP x) : data(x) {}

            coot::Mat<T> get() {
                // Check if it's a matrix
                if (!Rf_isMatrix(data)) {
                    // If it's a vector, convert to column matrix
                    return vector_to_mat();
                }

                // Get dimensions
                Shield<SEXP> dims(Rf_getAttrib(data, R_DimSymbol));
                const coot::uword n_rows = static_cast<coot::uword>(INTEGER(dims)[0]);
                const coot::uword n_cols = static_cast<coot::uword>(INTEGER(dims)[1]);

                // Ahead of every allocation, host and device alike: a guard
                // that fires after the truncated buffer exists is no guard.
                bandicoot_check_alloc<T>(static_cast<double>(n_rows) * static_cast<double>(n_cols),
                                         "Bandicoot matrix");

                // Allocate Bandicoot matrix
                coot::Mat<T> result(n_rows, n_cols);

                // Copy data from R to CPU memory
                std::vector<T> cpu_mem(n_rows * n_cols);
                bandicoot_copy_from_r(data, cpu_mem.data(), n_rows * n_cols, "Bandicoot matrix");

                // Copy from CPU to GPU
                result.copy_into_dev_mem(cpu_mem.data(), n_rows * n_cols);

                return result;
            }

        private:
            SEXP data;

            coot::Mat<T> vector_to_mat() {
                const coot::uword n_elem = Rf_length(data);

                bandicoot_check_alloc<T>(static_cast<double>(n_elem), "Bandicoot matrix");

                coot::Mat<T> result(n_elem, 1);

                std::vector<T> cpu_mem(n_elem);
                bandicoot_copy_from_r(data, cpu_mem.data(), n_elem, "Bandicoot matrix");
                result.copy_into_dev_mem(cpu_mem.data(), n_elem);

                return result;
            }
        };

        // Exporter for coot::Col<T> - convert R vector to Bandicoot column vector
        template <typename T>
        class Exporter< coot::Col<T> > {
        public:
            Exporter(SEXP x) : data(x) {}

            coot::Col<T> get() {
                coot::uword n_elem;

                // Handle both vectors and matrices
                if (Rf_isMatrix(data)) {
                    Shield<SEXP> dims(Rf_getAttrib(data, R_DimSymbol));
                    const coot::uword n_rows = static_cast<coot::uword>(INTEGER(dims)[0]);
                    const coot::uword n_cols = static_cast<coot::uword>(INTEGER(dims)[1]);

                    // Only accept column matrices (n_cols == 1)
                    if (n_cols != 1) {
                        Rcpp::stop("Cannot convert matrix with multiple columns to column vector");
                    }
                    n_elem = n_rows;
                } else {
                    n_elem = static_cast<coot::uword>(Rf_length(data));
                }

                bandicoot_check_alloc<T>(static_cast<double>(n_elem), "Bandicoot column vector");

                // Allocate Bandicoot column vector
                coot::Col<T> result(n_elem);

                // Copy data from R to CPU memory
                std::vector<T> cpu_mem(n_elem);
                bandicoot_copy_from_r(data, cpu_mem.data(), n_elem, "Bandicoot column vector");

                // Copy from CPU to GPU
                result.copy_into_dev_mem(cpu_mem.data(), n_elem);

                return result;
            }

        private:
            SEXP data;
        };

        // Exporter for coot::Row<T> - convert R vector to Bandicoot row vector
        template <typename T>
        class Exporter< coot::Row<T> > {
        public:
            Exporter(SEXP x) : data(x) {}

            coot::Row<T> get() {
                coot::uword n_elem;

                // Handle both vectors and matrices
                if (Rf_isMatrix(data)) {
                    Shield<SEXP> dims(Rf_getAttrib(data, R_DimSymbol));
                    const coot::uword n_rows = static_cast<coot::uword>(INTEGER(dims)[0]);
                    const coot::uword n_cols = static_cast<coot::uword>(INTEGER(dims)[1]);

                    // Only accept row matrices (n_rows == 1)
                    if (n_rows != 1) {
                        Rcpp::stop("Cannot convert matrix with multiple rows to row vector");
                    }
                    n_elem = n_cols;
                } else {
                    n_elem = static_cast<coot::uword>(Rf_length(data));
                }

                bandicoot_check_alloc<T>(static_cast<double>(n_elem), "Bandicoot row vector");

                // Allocate Bandicoot row vector
                coot::Row<T> result(n_elem);

                // Copy data from R to CPU memory
                std::vector<T> cpu_mem(n_elem);
                bandicoot_copy_from_r(data, cpu_mem.data(), n_elem, "Bandicoot row vector");

                // Copy from CPU to GPU
                result.copy_into_dev_mem(cpu_mem.data(), n_elem);

                return result;
            }

        private:
            SEXP data;
        };

        // Exporter for coot::Cube<T> - convert R 3D array to Bandicoot cube
        template <typename T>
        class Exporter< coot::Cube<T> > {
        public:
            Exporter(SEXP x) : data(x) {}

            coot::Cube<T> get() {
                // Check if it's an array with dimensions
                Shield<SEXP> dims(Rf_getAttrib(data, R_DimSymbol));

                if (Rf_isNull(dims) || Rf_length(dims) != 3) {
                    Rcpp::stop("Expected a 3-dimensional array for conversion to Bandicoot Cube");
                }

                // Get dimensions
                const coot::uword n_rows = static_cast<coot::uword>(INTEGER(dims)[0]);
                const coot::uword n_cols = static_cast<coot::uword>(INTEGER(dims)[1]);
                const coot::uword n_slices = static_cast<coot::uword>(INTEGER(dims)[2]);
                const coot::uword n_elem = n_rows * n_cols * n_slices;

                bandicoot_check_alloc<T>(static_cast<double>(n_rows) *
                                         static_cast<double>(n_cols) *
                                         static_cast<double>(n_slices),
                                         "Bandicoot cube");

                // Allocate Bandicoot cube
                coot::Cube<T> result(n_rows, n_cols, n_slices);

                // Copy data from R to CPU memory
                std::vector<T> cpu_mem(n_elem);
                bandicoot_copy_from_r(data, cpu_mem.data(), n_elem, "Bandicoot cube");

                // Copy from CPU to GPU
                result.copy_into_dev_mem(cpu_mem.data(), n_elem);

                return result;
            }

        private:
            SEXP data;
        };

    } // namespace traits
} // namespace Rcpp

#endif
