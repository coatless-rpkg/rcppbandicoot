// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// These statically-compiled definitions are available in any Bandicoot kernel.
typedef float2 cx_float;
#ifdef COOT_HAVE_FP64
typedef double2 cx_double;
#endif

#define COOT_FN2(ARG1,ARG2)  ARG1 ## ARG2
#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2)

#define COOT_FN_3_2(ARG1,ARG2,ARG3) ARG1 ## ARG2 ## ARG3
#define COOT_FN_3(ARG1,ARG2,ARG3) COOT_FN_3_2(ARG1,ARG2,ARG3)

// Sometimes we need to approximate Armadillo functionality that uses
// double---but double may not be available.  So we do our best...
#ifdef COOT_HAVE_FP64
  #define ARMA_FP_TYPE double
  #define ARMA_FP_MAX DBL_MAX
  #define ARMA_FP_MIN DBL_MIN
#else
  #define ARMA_FP_TYPE float
  #define ARMA_FP_MAX FLT_MAX
  #define ARMA_FP_MIN FLT_MIN
#endif
