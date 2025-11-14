// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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


// utility to turn a type into the string prefix that is used for that type

template<typename eT>
inline
std::string
type_prefix()
  {
  // We can't really use SFINAE because it's possible that uword (e.g. size_t) and u64/u32 are the same type.
  // But that's not guaranteed!
       if(is_same_type<eT,u8    >::yes)  { return "u8"; }
  else if(is_same_type<eT,s8    >::yes)  { return "s8"; }
  else if(is_same_type<eT,u16   >::yes)  { return "u16"; }
  else if(is_same_type<eT,s16   >::yes)  { return "s16"; }
  else if(is_same_type<eT,u32   >::yes)  { return "u32"; }
  else if(is_same_type<eT,s32   >::yes)  { return "s32"; }
  else if(is_same_type<eT,u64   >::yes)  { return "u64"; }
  else if(is_same_type<eT,s64   >::yes)  { return "s64"; }
  else if(is_same_type<eT,fp16  >::yes)  { return "h";   }
  else if(is_same_type<eT,float >::yes)  { return "f";   }
  else if(is_same_type<eT,double>::yes)  { return "d";   }
  else if(is_same_type<eT,uword >::yes)
    {
         if (sizeof(uword) == sizeof(u32)) { return "u32"; }
    else if (sizeof(uword) == sizeof(u64)) { return "u64"; }
    else                                   { return "?";   }
    }
  else if(is_same_type<eT,sword >::yes)
    {
         if (sizeof(sword) == sizeof(s32)) { return "s32"; }
    else if (sizeof(sword) == sizeof(s64)) { return "s64"; }
    else                                   { return "?";   }
    }
  else                                   { return "?";   }
  }
