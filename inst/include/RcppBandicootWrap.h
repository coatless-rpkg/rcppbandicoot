// RcppBandicootWrap.h: Wrap functions for Rcpp/Bandicoot glue
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

#ifndef RcppBandicoot__RcppBandicootWrap__h
#define RcppBandicoot__RcppBandicootWrap__h

#include <type_traits>

namespace Rcpp {
    namespace traits {

        // Maps a Bandicoot element type onto the R vector type that holds it.
        //
        // The integral cases are decided from sizeof and signedness rather than
        // from a specialization per spelling, because the spellings differ by
        // platform. coot::uword is std::size_t (typedef_elem.hpp), which is
        // 'unsigned long' on the LP64 Unixes but 'unsigned long long' on
        // Windows x64 (LLP64) -- a distinct type there, since 'unsigned long'
        // is only four bytes. Listing types by name therefore leaves
        // coot::uvec/umat with no mapping on Windows, and because the primary
        // template used to be declared but never defined that showed up as an
        // unresolved symbol at link time rather than as a compile error.
        template <typename T>
        struct bandicoot_sexptype {
            static_assert(std::is_integral<T>::value,
                          "RcppBandicoot: this Bandicoot element type has no corresponding R "
                          "vector type. Supported element types are float, double, "
                          "std::complex<float>, std::complex<double>, Rbyte, and integral types.");

            // R's only integer vector is INTSXP: 32-bit signed, with INT_MIN
            // reserved for NA. So a signed type of at most four bytes round
            // trips through it, and an unsigned type only if it is narrower
            // than four bytes. Everything wider goes to REALSXP, which is how R
            // itself carries integers it cannot fit in an integer vector; a
            // double is exact for every value up to 2^53.
            static const int value =
                (std::is_signed<T>::value ? (sizeof(T) <= 4) : (sizeof(T) < 4)) ? INTSXP : REALSXP;
        };

        template <> struct bandicoot_sexptype<double>                { static const int value = REALSXP; };
        template <> struct bandicoot_sexptype<float>                 { static const int value = REALSXP; };
        template <> struct bandicoot_sexptype<Rbyte>                 { static const int value = RAWSXP;  };
        template <> struct bandicoot_sexptype<std::complex<double> > { static const int value = CPLXSXP; };
        template <> struct bandicoot_sexptype<std::complex<float> >  { static const int value = CPLXSXP; };

        // Helper function to get R SEXP type from C++ type
        template <typename T> inline int bandicoot_get_sexptype() { return bandicoot_sexptype<T>::value; }

        // Element copy into the R vector that bandicoot_sexptype<T> selected.
        //
        // The destination has to be picked at compile time. Branching on the
        // SEXP type at run time compiles every branch for every T, so the
        // REALSXP branch -- std::copy from a std::complex<double> into a
        // double* -- makes wrap() ill-formed for the complex element types even
        // though a CPLXSXP mapping exists for them. Dispatching on a tag
        // carrying the SEXP type instantiates only the branch valid for T.
        template <int RTYPE> struct bandicoot_sexptype_tag {};

        template <typename T>
        inline void bandicoot_copy_to_r(const std::vector<T>& src, SEXP dest, bandicoot_sexptype_tag<REALSXP>) {
            std::copy(src.begin(), src.end(), REAL(dest));
        }

        template <typename T>
        inline void bandicoot_copy_to_r(const std::vector<T>& src, SEXP dest, bandicoot_sexptype_tag<INTSXP>) {
            int* r_ptr = INTEGER(dest);
            for (coot::uword i = 0; i < src.size(); ++i) {
                r_ptr[i] = static_cast<int>(src[i]);
            }
        }

        template <typename T>
        inline void bandicoot_copy_to_r(const std::vector<T>& src, SEXP dest, bandicoot_sexptype_tag<RAWSXP>) {
            std::copy(src.begin(), src.end(), RAW(dest));
        }

        template <typename T>
        inline void bandicoot_copy_to_r(const std::vector<T>& src, SEXP dest, bandicoot_sexptype_tag<CPLXSXP>) {
            Rcomplex* r_ptr = COMPLEX(dest);
            for (coot::uword i = 0; i < src.size(); ++i) {
                r_ptr[i].r = static_cast<double>(std::real(src[i]));
                r_ptr[i].i = static_cast<double>(std::imag(src[i]));
            }
        }

        template <typename T>
        inline void bandicoot_copy_to_r(const std::vector<T>& src, SEXP dest) {
            bandicoot_copy_to_r(src, dest, bandicoot_sexptype_tag<bandicoot_sexptype<T>::value>());
        }

    } // namespace traits

    // wrap for coot::Mat<T> - matrix
    template <typename T>
    inline SEXP wrap(const coot::Mat<T>& x) {
        const int RTYPE = traits::bandicoot_get_sexptype<T>();
        const coot::uword n_rows = x.n_rows;
        const coot::uword n_cols = x.n_cols;

        // Allocate R matrix
        SEXP res = PROTECT(Rf_allocMatrix(RTYPE, n_rows, n_cols));

        // Copy data from GPU to CPU to R
        // Bandicoot matrices are stored in column-major order, same as R
        std::vector<T> cpu_mem(n_rows * n_cols);
        x.copy_from_dev_mem(cpu_mem.data(), n_rows * n_cols);

        // Copy to R object
        traits::bandicoot_copy_to_r(cpu_mem, res);

        UNPROTECT(1);
        return res;
    }

    // wrap for coot::Col<T> - column vector
    template <typename T>
    inline SEXP wrap(const coot::Col<T>& x) {
        const int RTYPE = traits::bandicoot_get_sexptype<T>();
        const coot::uword n_elem = x.n_elem;

        // Allocate R vector
        SEXP res = PROTECT(Rf_allocVector(RTYPE, n_elem));

        // Copy data from GPU to CPU to R
        std::vector<T> cpu_mem(n_elem);
        x.copy_from_dev_mem(cpu_mem.data(), n_elem);

        // Copy to R object
        traits::bandicoot_copy_to_r(cpu_mem, res);

        UNPROTECT(1);
        return res;
    }

    // wrap for coot::Row<T> - row vector
    template <typename T>
    inline SEXP wrap(const coot::Row<T>& x) {
        const int RTYPE = traits::bandicoot_get_sexptype<T>();
        const coot::uword n_elem = x.n_elem;

        // Allocate R vector (R doesn't distinguish between row and column vectors)
        SEXP res = PROTECT(Rf_allocVector(RTYPE, n_elem));

        // Copy data from GPU to CPU to R
        std::vector<T> cpu_mem(n_elem);
        x.copy_from_dev_mem(cpu_mem.data(), n_elem);

        // Copy to R object
        traits::bandicoot_copy_to_r(cpu_mem, res);

        UNPROTECT(1);
        return res;
    }

    // wrap for coot::Cube<T> - 3D array
    template <typename T>
    inline SEXP wrap(const coot::Cube<T>& x) {
        const int RTYPE = traits::bandicoot_get_sexptype<T>();
        const coot::uword n_rows = x.n_rows;
        const coot::uword n_cols = x.n_cols;
        const coot::uword n_slices = x.n_slices;
        const coot::uword n_elem = x.n_elem;

        // Allocate R 3D array
        SEXP res = PROTECT(Rf_alloc3DArray(RTYPE, n_rows, n_cols, n_slices));

        // Copy data from GPU to CPU to R
        std::vector<T> cpu_mem(n_elem);
        x.copy_from_dev_mem(cpu_mem.data(), n_elem);

        // Copy to R object
        traits::bandicoot_copy_to_r(cpu_mem, res);

        UNPROTECT(1);
        return res;
    }

    // wrap for coot::subview<T> - matrix subview
    template <typename T>
    inline SEXP wrap(const coot::subview<T>& x) {
        // Convert to Mat first, then wrap
        coot::Mat<T> tmp = x;
        return wrap(tmp);
    }

    // wrap for coot::subview_col<T> - column subview
    template <typename T>
    inline SEXP wrap(const coot::subview_col<T>& x) {
        // Convert to Col first, then wrap
        coot::Col<T> tmp = x;
        return wrap(tmp);
    }

    // wrap for coot::subview_row<T> - row subview
    template <typename T>
    inline SEXP wrap(const coot::subview_row<T>& x) {
        // Convert to Row first, then wrap
        coot::Row<T> tmp = x;
        return wrap(tmp);
    }

    // wrap for coot::diagview<T> - diagonal view
    template <typename T>
    inline SEXP wrap(const coot::diagview<T>& x) {
        // Convert to Col first, then wrap
        coot::Col<T> tmp = x;
        return wrap(tmp);
    }

} // namespace Rcpp

#endif
