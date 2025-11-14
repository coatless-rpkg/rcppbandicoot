// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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



// Implementations of the variable philox algorithm to generate random numbers.
// Adapted from Mitchell, Stokes, Frank, and Holmes (2022), Listing 1.



inline
UWORD
var_philox(const UWORD val, const __global UWORD* keys, const unsigned char bits)
  {
  // via Salmon, Moraes, Dror, and Shaw (2011): "Parallel random numbers: as easy as 1, 2, 3".
  const UWORD M0 = 0xD2B74407B1CE6E93;

  // The right side is allowed to have the extra bits.
  const unsigned char right_side_bits = (bits + 1) / 2;
  const unsigned char left_side_bits = bits / 2;
  const uint left_mask  = (((uint) 1) << left_side_bits) - 1;
  const uint right_mask = (((uint) 1) << right_side_bits) - 1;

  uint state0 = (uint) (val >> right_side_bits);
  uint state1 = (uint) (val & right_mask);

  // 24 rounds is what is needed to pass all the RNG tests (see section 5 of the paper).
  uint hi, lo;
  for (unsigned char i = 0; i < 24; ++i)
    {

    // 64-bit integer multiplication, split the results into two uints
    UWORD hilo = M0 * state0;
    hi = (hilo >> 32);
    lo = (uint) hilo;

    lo = (lo << (right_side_bits - left_side_bits)) | (state1 >> left_side_bits);

    state0 = ((hi ^ keys[i]) ^ state1) & left_mask;
    state1 = lo & right_mask;
    }

  // Combine the sides for the result.
  return ((UWORD) (state0 << right_side_bits)) | ((UWORD) state1);
  }
