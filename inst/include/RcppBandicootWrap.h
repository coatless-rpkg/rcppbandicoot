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

namespace Rcpp {
    namespace traits {
        // Helper template to get R SEXP type from C++ type
        template <typename T> inline int bandicoot_get_sexptype();

        template <> inline int bandicoot_get_sexptype<double>()                { return REALSXP; }
        template <> inline int bandicoot_get_sexptype<float>()                 { return REALSXP; }
        template <> inline int bandicoot_get_sexptype<int>()                   { return INTSXP; }
        template <> inline int bandicoot_get_sexptype<unsigned int>()          { return INTSXP; }
        template <> inline int bandicoot_get_sexptype<long>()                  { return REALSXP; }
        template <> inline int bandicoot_get_sexptype<unsigned long>()         { return REALSXP; }
        // Skip std::size_t as its specialized above (unsigned int, unsigned long, etc.)
        template <> inline int bandicoot_get_sexptype<Rbyte>()                 { return RAWSXP; }
        template <> inline int bandicoot_get_sexptype<std::complex<double>>()  { return CPLXSXP; }
        template <> inline int bandicoot_get_sexptype<std::complex<float>>()   { return CPLXSXP; }
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
        if (RTYPE == REALSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), REAL(res));
        } else if (RTYPE == INTSXP) {
            int* r_ptr = INTEGER(res);
            for (coot::uword i = 0; i < cpu_mem.size(); ++i) {
                r_ptr[i] = static_cast<int>(cpu_mem[i]);
            }
        } else if (RTYPE == CPLXSXP) {
            Rcomplex* r_ptr = COMPLEX(res);
            for (coot::uword i = 0; i < cpu_mem.size(); ++i) {
                r_ptr[i].r = static_cast<double>(std::real(cpu_mem[i]));
                r_ptr[i].i = static_cast<double>(std::imag(cpu_mem[i]));
            }
        } else if (RTYPE == RAWSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), RAW(res));
        }

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
        if (RTYPE == REALSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), REAL(res));
        } else if (RTYPE == INTSXP) {
            int* r_ptr = INTEGER(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i] = static_cast<int>(cpu_mem[i]);
            }
        } else if (RTYPE == CPLXSXP) {
            Rcomplex* r_ptr = COMPLEX(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i].r = static_cast<double>(std::real(cpu_mem[i]));
                r_ptr[i].i = static_cast<double>(std::imag(cpu_mem[i]));
            }
        } else if (RTYPE == RAWSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), RAW(res));
        }

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
        if (RTYPE == REALSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), REAL(res));
        } else if (RTYPE == INTSXP) {
            int* r_ptr = INTEGER(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i] = static_cast<int>(cpu_mem[i]);
            }
        } else if (RTYPE == CPLXSXP) {
            Rcomplex* r_ptr = COMPLEX(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i].r = static_cast<double>(std::real(cpu_mem[i]));
                r_ptr[i].i = static_cast<double>(std::imag(cpu_mem[i]));
            }
        } else if (RTYPE == RAWSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), RAW(res));
        }

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
        if (RTYPE == REALSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), REAL(res));
        } else if (RTYPE == INTSXP) {
            int* r_ptr = INTEGER(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i] = static_cast<int>(cpu_mem[i]);
            }
        } else if (RTYPE == CPLXSXP) {
            Rcomplex* r_ptr = COMPLEX(res);
            for (coot::uword i = 0; i < n_elem; ++i) {
                r_ptr[i].r = static_cast<double>(std::real(cpu_mem[i]));
                r_ptr[i].i = static_cast<double>(std::imag(cpu_mem[i]));
            }
        } else if (RTYPE == RAWSXP) {
            std::copy(cpu_mem.begin(), cpu_mem.end(), RAW(res));
        }

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
