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

// See algorithm "xorwow" from page 5 of "Xorshift RNGs" by George Marsaglia.
inline
uint
xorwow_rng_uint(uint* xorwow_state)
  {
  // xorwow_state[0] through xorwow_state[4] represent the 5 state integers,
  // and xorwow_state[5] holds the counter.
  uint t = xorwow_state[4] ^ (xorwow_state[4] >> 2);

  xorwow_state[4] = xorwow_state[3];
  xorwow_state[3] = xorwow_state[2];
  xorwow_state[2] = xorwow_state[1];
  xorwow_state[1] = xorwow_state[0];
  xorwow_state[0] ^= (xorwow_state[0] << 4) ^ (t ^ (t << 1));

  // Following Saito and Matsumoto (2012), we use a larger constant for d so that the higher bits flip more often.
  // We ignore their conclusion that XORWOW has problems (it's fast!).
  xorwow_state[5] += 268183997;
  return xorwow_state[0] + xorwow_state[5];
  }



// See algorithm "xorwow" from page 5 of "Xorshift RNGs" by George Marsaglia.
inline
ulong
xorwow_rng_ulong(ulong* xorwow_state)
  {
  // xorwow_state[0] through xorwow_state[4] represent the 5 state integers,
  // and xorwow_state[5] holds the counter.
  ulong t = xorwow_state[4] ^ (xorwow_state[4] >> 2);

  xorwow_state[4] = xorwow_state[3];
  xorwow_state[3] = xorwow_state[2];
  xorwow_state[2] = xorwow_state[1];
  xorwow_state[1] = xorwow_state[0];
  xorwow_state[0] ^= (xorwow_state[0] << 4) ^ (t ^ (t << 1));

  // Following Saito and Matsumoto (2012), we use a larger constant for d so that the higher bits flip more often.
  // We ignore their conclusion that XORWOW has problems (it's fast!).
  xorwow_state[5] += 2274084621458550325;
  return xorwow_state[0] + xorwow_state[5];
  }
