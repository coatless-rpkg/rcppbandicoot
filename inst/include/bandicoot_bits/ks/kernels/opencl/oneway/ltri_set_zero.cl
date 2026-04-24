// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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

__kernel
void
COOT_FN(PREFIX,ltri_set_zero)(__global eT1* out,
                              const UWORD out_offset,
                              const UWORD n_rows,
                              const UWORD n_cols)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD index = row + n_rows * col;
  if( (row < n_rows) && (col < n_cols) && (row > col) )
    {
    out[index + out_offset] = (eT1)(0);
    }
  }
