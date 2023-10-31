// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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

__global__
void
COOT_FN(PREFIX,equ_array_sinc_post)(eT2* dest,
                                    const eT1* src,
                                    const eT1 val_pre,
                                    const eT2 val_post,
                                    const UWORD n_rows,
                                    const UWORD n_cols,
                                    const UWORD dest_M_n_rows,
                                    const UWORD src_M_n_rows)
  {
  (void) (val_pre);
  (void) (val_post);

  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  const UWORD src_index = row + col * src_M_n_rows;
  const UWORD dest_index = row + col * dest_M_n_rows;

  if (row < n_rows && col < n_cols)
    {
    const eT1 val = (eT1) src[src_index];
    // To imitate Armadillo correctly, we use double if the type is not floating point.
    if (coot_is_fp(val))
      {
      const fp_eT1 tmp = val * COOT_PI;
      dest[dest_index] = (tmp == (eT1) 0.0) ? (eT2) 1.0 : (eT2) (sin(tmp) / tmp);
      }
    else
      {
      const double fp_val = (double) val;
      const double tmp = fp_val * COOT_PI;
      dest[dest_index] = (tmp == 0.0) ? (eT2) 1.0 : (eT2) (sin(tmp) / tmp);
      }
    }
  }
