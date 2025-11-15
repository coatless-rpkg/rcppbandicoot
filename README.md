# RcppBandicoot

<!-- badges: start -->
[![R-CMD-check](https://github.com/coatless/rcppbandicoot-priv/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/coatless/rcppbandicoot-priv/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

## Overview

[Bandicoot](https://coot.sourceforge.io/) is a C++ header-only GPU accelerated
linear algebra library written by developers behind [Armadillo](https://arma.sourceforge.io/) that provides high-level syntax for performing computations on
graphics processing units.

The RcppBandicoot package includes the header files from the Bandicoot library
and integrates a seamless experience with R by using the 
[Rcpp](https://cran.r-project.org/package=Rcpp) package. 
Therefore, users do not need to install Bandicoot to use RcppBandicoot. 

## Installation

You can install RcppBandicoot from GitHub:

```r
# install.packages("remotes")
remotes::install_github("coatless/rcppbandicoot")
```

## Usage

To use RcppBandicoot in your package, add the following to your `DESCRIPTION` file:

```
LinkingTo: Rcpp, RcppBandicoot
Imports: Rcpp
```

Then in your C++ code, include the main header:

```cpp
#include <RcppBandicoot.h>

// [[Rcpp::depends(RcppBandicoot)]]

// [[Rcpp::export]]
coot::mat gpu_multiply(const coot::mat& A, const coot::mat& B) {
    return A * B;  // Computed on GPU
}
```

The RcppBandicoot integration provides automatic conversion between:
- R matrices ↔ `coot::Mat<T>`
- R vectors ↔ `coot::Col<T>` and `coot::Row<T>`
- R 3D arrays ↔ `coot::Cube<T>`

All computations are performed on the GPU for maximum performance.

## Status

The package is under active development with releases to [CRAN](https://cran.r-project.org/)
about once a month.

## Requirements

- **C++14 compatible compiler** (required by Bandicoot library)
- OpenCL 1.2+ or CUDA 9.8+ for GPU support
- GPU device with appropriate drivers
- CLBlast (recommended) or clBLAS for OpenCL BLAS operations

### Authors

James Joseph Balamuta

### License

GPL (>= 2)
