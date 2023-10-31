// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2019 Ryan Curtin (https://www.ratml.org)
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



template<typename T1>
coot_hot
inline
void
coot_check_cl_error(const cl_int error_code, const T1& x)
  {
  if(error_code != CL_SUCCESS)
    {
    coot_stop_runtime_error( x, coot_cl_error::as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_clblas_error(const cl_int error_code, const T1& x)
  {
  if(error_code != CL_SUCCESS)
    {
    coot_stop_runtime_error( x, coot_clblas_error::as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_magma_error(const magma_int_t error_code, const T1& x)
  {
  if (error_code != 0)
    {
    coot_stop_runtime_error( x, magma::error_as_string(error_code) );
    }
  }
