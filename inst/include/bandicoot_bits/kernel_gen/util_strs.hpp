// Copyright 2026 Ryan Curtin (http://www.ratml.org)
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



//
// utility string constants used during macro assembly
//



// simple separators and operators

struct empty_str           { static inline constexpr auto& str() { return "";      } };
struct open_paren          { static inline constexpr auto& str() { return "(";     } };
struct double_open_paren   { static inline constexpr auto& str() { return "((";    } };
struct close_paren         { static inline constexpr auto& str() { return ")";     } };
struct space               { static inline constexpr auto& str() { return " ";     } };
struct space_sep           { static inline constexpr auto& str() { return ", ";    } };
struct space_semicolon     { static inline constexpr auto& str() { return "; ";    } };
struct spaced_lt           { static inline constexpr auto& str() { return " < ";   } };
struct spaced_plus         { static inline constexpr auto& str() { return " + ";   } };
struct spaced_mod          { static inline constexpr auto& str() { return " % ";   } };
struct spaced_mul          { static inline constexpr auto& str() { return " * ";   } };
struct spaced_div          { static inline constexpr auto& str() { return " / ";   } };
struct double_close_paren  { static inline constexpr auto& str() { return "))";    } };
struct triple_close_paren  { static inline constexpr auto& str() { return ")))";   } };
struct spaced_and          { static inline constexpr auto& str() { return " && ";  } };
struct double_arr_close    { static inline constexpr auto& str() { return "))]";   } };
struct close_space_sep     { static inline constexpr auto& str() { return "), ";   } };
struct space_x             { static inline constexpr auto& str() { return " x";    } };
struct space_y             { static inline constexpr auto& str() { return " y";    } };
struct paren_x             { static inline constexpr auto& str() { return "(x)";   } };
struct semicolon_close     { static inline constexpr auto& str() { return "; }\n"; } };
struct underscore          { static inline constexpr auto& str() { return "_";     } };
struct equals              { static inline constexpr auto& str() { return "=";     } };



// auxiliary names used in multiple contexts


struct coot_object_prefix  { static inline constexpr auto& str() { return "COOT_OBJECT_";            } };
struct coot_name_arg       { static inline constexpr auto& str() { return "(name)=";                 } };
struct uword_arg_prefix    { static inline constexpr auto& str() { return "UWORD COOT_CONCAT(name,"; } };
struct et_arg_name         { static inline constexpr auto& str() { return " COOT_CONCAT(name,";      } };
struct coot_concat_name    { static inline constexpr auto& str() { return "COOT_CONCAT(name,";       } };




// names used for auxiliary parameters for kernels

struct n_rows_name            { static inline constexpr auto& str() { return "_n_rows";         } };
struct n_cols_name            { static inline constexpr auto& str() { return "_n_cols";         } };
struct n_slices_name          { static inline constexpr auto& str() { return "_n_slices";       } };
struct n_elem_name            { static inline constexpr auto& str() { return "_n_elem";         } };
struct M_n_rows_name          { static inline constexpr auto& str() { return "_M_n_rows";       } };
struct M_n_elem_slice_name    { static inline constexpr auto& str() { return "_M_n_elem_slice"; } };
struct incr_name              { static inline constexpr auto& str() { return "_incr";           } };
struct elem1_index_prefix     { static inline constexpr auto& str() { return "_index";          } };
struct elem2_src_prefix       { static inline constexpr auto& str() { return "_src";            } };
struct elem2_row_index_prefix { static inline constexpr auto& str() { return "_row_index";      } };
struct elem2_col_index_prefix { static inline constexpr auto& str() { return "_col_index";      } };
struct target_name            { static inline constexpr auto& str() { return "_target";         } };
struct arg_prefix_name        { static inline constexpr auto& str() { return "_arg";            } };
struct eop_aux1_name          { static inline constexpr auto& str() { return "_aux";            } };
struct eop_aux2_name          { static inline constexpr auto& str() { return "_aux2";           } };
struct eglue_arg1_name        { static inline constexpr auto& str() { return "_a";              } };
struct eglue_arg2_name        { static inline constexpr auto& str() { return "_b";              } };
struct eop_scalar_arg_name    { static inline constexpr auto& str() { return "a";               } };



// literals used for Vulkan buffer construction

struct vk_buf_prefix    { static inline constexpr auto& str() { return "layout(set=0, binding=";            } };
struct vk_buf_std430    { static inline constexpr auto& str() { return ", std430) buffer Buf";              } };
struct vk_buf_et_prefix { static inline constexpr auto& str() { return " { ET";                             } };
struct vk_buf_suffix    { static inline constexpr auto& str() { return " data[]; } COOT_CONCAT(name,_buf)"; } };
struct params_suffix    { static inline constexpr auto& str() { return "_PARAMS(name)=";                    } };




// used during bounds checks and element access

struct bounds_check_arg_macro { static inline constexpr auto& str() { return "_BOUNDS_CHECK(name, row, col, slice)="; } };
struct at_arg_macro           { static inline constexpr auto& str() { return "_AT(name, row, col, slice)=";           } };
struct macro_arg1             { static inline constexpr auto& str() { return "row";                                   } };
struct macro_arg2             { static inline constexpr auto& str() { return "col";                                   } };
struct macro_arg3             { static inline constexpr auto& str() { return "slice";                                 } };
struct conj_str               { static inline constexpr auto& str() { return "coot_conj";                             } };



// names used for function construction

struct eop_inline_function_arg  { static inline constexpr auto& str() { return "(const ";     } };
struct eop_inline_function_body { static inline constexpr auto& str() { return ") { return "; } };
struct eop_extra_arg_list_const { static inline constexpr auto& str() { return ", const ";    } };




//
//
