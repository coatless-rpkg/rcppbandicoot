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

#ifndef RcppBandicoot__RcppBandicoot__h
#define RcppBandicoot__RcppBandicoot__h

// Prevent direct inclusion of Rcpp.h when using RcppBandicoot
#if defined(Rcpp_hpp) && !defined(COMPILING_RCPPBANDICOOT)
 #error "The file 'Rcpp.h' should not be included. Please correct to include only 'RcppBandicoot.h'."
#endif

// Step 1: Include forward declarations and load Bandicoot library
// This must be done before including Rcpp.h
#include <RcppBandicootForward.h>

// Step 2: Now include Rcpp
#include <Rcpp/Rcpp>

// Step 3: Include the conversion functions (wrap and as)
#include <RcppBandicootWrap.h>
#include <RcppBandicootAs.h>

#endif