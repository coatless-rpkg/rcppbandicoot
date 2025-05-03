// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://www.ratml.org)
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



template<typename eT>
inline
bool
diskio::convert_token(eT& val, const std::string& token)
  {
  const size_t N = size_t(token.length());

  const char* str = token.c_str();

  if( (N == 0) || ((N == 1) && (str[0] == '0')) )  { val = eT(0); return true; }

  if( (N == 3) || (N == 4) )
    {
    const bool neg = (str[0] == '-');
    const bool pos = (str[0] == '+');

    const size_t offset = ( (neg || pos) && (N == 4) ) ? 1 : 0;

    const char sig_a = str[offset  ];
    const char sig_b = str[offset+1];
    const char sig_c = str[offset+2];

    if( ((sig_a == 'i') || (sig_a == 'I')) && ((sig_b == 'n') || (sig_b == 'N')) && ((sig_c == 'f') || (sig_c == 'F')) )
      {
      val = neg ? cond_rel< is_signed<eT>::value >::make_neg(Datum<eT>::inf) : Datum<eT>::inf;

      return true;
      }
    else
    if( ((sig_a == 'n') || (sig_a == 'N')) && ((sig_b == 'a') || (sig_b == 'A')) && ((sig_c == 'n') || (sig_c == 'N')) )
      {
      val = Datum<eT>::nan;

      return true;
      }
    }
  char* endptr = nullptr;

  if(is_real<eT>::value)
    {
    val = eT( std::strtod(str, &endptr) );
    }
  else
    {
    if(is_signed<eT>::value)
      {
      // signed integer

      val = eT( std::strtoll(str, &endptr, 10) );
      }
    else
      {
      // unsigned integer

      if((str[0] == '-') && (N >= 2))
        {
        val = eT(0);

        if((str[1] == '-') || (str[1] == '+')) { return false; }

        const char* str_offset1 = &(str[1]);

        std::strtoull(str_offset1, &endptr, 10);

        if(str_offset1 == endptr)  { return false; }

        return true;
        }

      val = eT( std::strtoull(str, &endptr, 10) );
      }
    }

  if(str == endptr)  { return false; }

  return true;
  }



template<typename T>
inline
bool
diskio::convert_token(std::complex<T>& val, const std::string& token)
  {
  const size_t N   = size_t(token.length());
  const size_t Nm1 = N-1;

  if(N == 0)  { val = std::complex<T>(0); return true; }

  const char* str = token.c_str();

  // valid complex number formats:
  // (real,imag)
  // (real)
  // ()

  if( (token[0] != '(') || (token[Nm1] != ')') )
    {
    // no brackets, so treat the token as a non-complex number

    T val_real;

    const bool state = diskio::convert_token(val_real, token);  // use the non-complex version of this function

    val = std::complex<T>(val_real);

    return state;
    }

  // does the token contain only the () brackets?
  if(N <= 2)  { val = std::complex<T>(0); return true; }

  size_t comma_loc   = 0;
  bool   comma_found = false;

  for(size_t i=0; i<N; ++i)  { if(str[i] == ',')  { comma_loc = i; comma_found = true; break; } }

  bool state = false;

  if(comma_found == false)
    {
    // only the real part is available

    const std::string token_real( &(str[1]), (Nm1 - 1) );

    T val_real;

    state = diskio::convert_token(val_real, token_real);  // use the non-complex version of this function

    val = std::complex<T>(val_real);
    }
  else
    {
    const std::string token_real( &(str[1]),           (comma_loc - 1      ) );
    const std::string token_imag( &(str[comma_loc+1]), (Nm1 - 1 - comma_loc) );

    T val_real;
    T val_imag;

    const bool state_real = diskio::convert_token(val_real, token_real);
    const bool state_imag = diskio::convert_token(val_imag, token_imag);

    state = (state_real && state_imag);

    val = std::complex<T>(val_real, val_imag);
    }

  return state;
  }
