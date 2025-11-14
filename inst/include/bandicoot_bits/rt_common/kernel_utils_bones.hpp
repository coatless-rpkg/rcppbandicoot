// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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

// utility functions required by all backends
// all in the coot::rt_common:: namespace


// Get the name of the kernel with the type prefix substituted in.
                                                   std::string get_kernel_name(const zeroway_kernel_id::enum_id num);
template<typename eT>                              std::string get_kernel_name(const oneway_kernel_id::enum_id num);
template<typename eT>                              std::string get_kernel_name(const oneway_real_kernel_id::enum_id num);
template<typename eT>                              std::string get_kernel_name(const oneway_integral_kernel_id::enum_id num);
template<typename eT1, typename eT2>               std::string get_kernel_name(const twoway_kernel_id::enum_id num);
template<typename eT1, typename eT2, typename eT3> std::string get_kernel_name(const threeway_kernel_id::enum_id num);
