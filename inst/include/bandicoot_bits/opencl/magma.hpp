// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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



inline std::string error_as_string(const magma_int_t error_code)
  {
  switch (error_code)
    {
    case MAGMA_SUCCESS:               return "MAGMA_SUCCESS (no error)";
    case MAGMA_ERR:                   return "MAGMA_ERR (unspecified error)";
    case MAGMA_ERR_NOT_INITIALIZED:   return "MAGMA_ERR_NOT_INITIALIZED";
    case MAGMA_ERR_NOT_SUPPORTED:     return "MAGMA_ERR_NOT_SUPPORTED";
    case MAGMA_ERR_NOT_FOUND:         return "MAGMA_ERR_NOT_FOUND (file not found)";
    case MAGMA_ERR_HOST_ALLOC:        return "MAGMA_ERR_HOST_ALLOC (could not allocate CPU host memory)";
    case MAGMA_ERR_DEVICE_ALLOC:      return "MAGMA_ERR_DEVICE_ALLOC (could not allocate device memory)";
    case MAGMA_ERR_INVALID_PTR:       return "MAGMA_ERR_INVALID_PTR (can't free invalid pointer)";
    case MAGMA_ERR_UNKNOWN:           return "MAGMA_ERR_UNKNOWN (unspecified error)";
    case MAGMA_ERR_NOT_IMPLEMENTED:   return "MAGMA_ERR_NOT_IMPLEMENTED";
    case MAGMA_ERR_NAN:               return "MAGMA_ERR_NAN (NaN detected)";
    default:
      {
      std::ostringstream oss;
      oss << "unspecified MAGMA error code (" << error_code << ")";
      return oss.str();
      }
    }
  }
