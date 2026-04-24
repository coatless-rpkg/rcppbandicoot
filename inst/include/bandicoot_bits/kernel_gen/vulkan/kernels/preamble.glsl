// Copyright 2026 Marcus Edel (http://www.kurg.org/)
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



R"(
#version 450
#ifdef COOT_USE_INT64
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

#define COOT_FN2(ARG1, ARG2) ARG1 ## ARG2
#define COOT_FN(ARG1, ARG2) COOT_FN2(ARG1, ARG2)
#define COOT_CONCAT(ARG1, ARG2) COOT_FN2(ARG1, ARG2)

#ifndef UWORD
#define UWORD uint64_t
#endif

)"
