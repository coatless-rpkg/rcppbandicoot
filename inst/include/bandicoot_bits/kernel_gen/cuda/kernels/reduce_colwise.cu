// Copyright 2026 Ryan Curtin (http://www.ratml.org)
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

__global__
void
(COOT_KERNEL_FUNC)(COOT_OBJECT_0(dest),
                   COOT_OBJECT_1(src))
  {
  const UWORD col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < COOT_CONCAT(src, _n_cols))
    {
    ET0 acc;
    COOT_INIT_OP(acc, COOT_TO_ET0(COOT_OBJECT_1_AT(src, 0, col, 0)));

    for (UWORD i = 1; i < COOT_CONCAT(src, _n_rows); ++i)
      {
      COOT_INNER_OP(acc, COOT_TO_ET0(COOT_OBJECT_1_AT(src, i, col, 0)));
      }

    COOT_OBJECT_0_AT(dest, col, 0, 0) = COOT_FINAL_OP(acc, COOT_CONCAT(src, _n_rows));
    }
  }

)"
