// RcppBandicootForward.h: Forward declarations for Rcpp/Bandicoot glue
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

#ifndef RcppBandicoot__RcppBandicootForward__h
#define RcppBandicoot__RcppBandicootForward__h

// Include RcppCommon and Rconfig first to get R's output mechanisms
#include <RcppCommon.h>
#include <Rconfig.h>

// Redirect Bandicoot output streams to R's output mechanism
// Use Rcpp's output streams instead of std::cout/std::cerr for R package compliance
#if !defined(COOT_COUT_STREAM)
#define COOT_COUT_STREAM Rcpp::Rcout
#endif
#if !defined(COOT_CERR_STREAM)
#define COOT_CERR_STREAM Rcpp::Rcerr
#endif

// Include the Bandicoot library
#include <bandicoot>

// Fix for macOS: OpenCL framework headers redefine TRUE/FALSE as plain integers,
// which conflicts with R's Rboolean type. Restore R's definitions.
#ifdef __APPLE__
#undef TRUE
#undef FALSE
#define TRUE (Rboolean)1
#define FALSE (Rboolean)0
#endif


// Forward declarations for Rcpp integration
// These need to be declared before including Rcpp.h

// Forward declare wrap functions in Rcpp namespace
namespace Rcpp {
    // Mat<T> - Matrix
    template <typename T> SEXP wrap(const coot::Mat<T>& x);

    // Col<T> - Column vector
    template <typename T> SEXP wrap(const coot::Col<T>& x);

    // Row<T> - Row vector
    template <typename T> SEXP wrap(const coot::Row<T>& x);

    // Cube<T> - 3D array
    template <typename T> SEXP wrap(const coot::Cube<T>& x);

    // subview<T> - Matrix subview
    template <typename T> SEXP wrap(const coot::subview<T>& x);

    // subview_col<T> - Column subview
    template <typename T> SEXP wrap(const coot::subview_col<T>& x);

    // subview_row<T> - Row subview
    template <typename T> SEXP wrap(const coot::subview_row<T>& x);

    // diagview<T> - Diagonal view
    template <typename T> SEXP wrap(const coot::diagview<T>& x);
} // namespace Rcpp

// Forward declare Exporter for Bandicoot types (as conversions) in Rcpp::traits namespace
namespace Rcpp {
    namespace traits {
        // Mat<T>
        template <typename T> class Exporter< coot::Mat<T> >;

        // Col<T>
        template <typename T> class Exporter< coot::Col<T> >;

        // Row<T>
        template <typename T> class Exporter< coot::Row<T> >;

        // Cube<T>
        template <typename T> class Exporter< coot::Cube<T> >;
    } // namespace traits
} // namespace Rcpp

#endif
