// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2022 Ryan Curtin (https://www.ratml.org/)
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

// This contains caching utilities for saving and loading compiled kernels.

// If in write mode, we will create the file (and directories if needed).
inline std::fstream open_cache(const std::string& unique_host_device_id, const bool write);

// Try to open `f` to read or write the file dirname/filename.
// If `write` is true, necessary directories will be created if possible.
inline void try_open(std::fstream& f,
                     const std::string& dirname,
                     const std::string& filename,
                     const bool write);

inline bool try_recursive_mkdir(const std::string& dirname);

inline size_t has_cached_kernels(const std::string& unique_host_device_id);

inline bool read_cached_kernels(const std::string& unique_host_device_id,
                                unsigned char* buffer);

inline bool cache_kernels(const std::string& unique_host_device_id,
                          const unsigned char* buffer,
                          const size_t buf_len);
