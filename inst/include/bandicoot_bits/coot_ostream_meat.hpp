// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



inline
coot_ostream_state::coot_ostream_state(const std::ostream& o)
  : orig_flags    (o.flags())
  , orig_precision(o.precision())
  , orig_width    (o.width())
  , orig_fill     (o.fill())
  {
  }



inline
void
coot_ostream_state::restore(std::ostream& o) const
  {
  o.flags    (orig_flags);
  o.precision(orig_precision);
  o.width    (orig_width);
  o.fill     (orig_fill);
  }



//
//



template<typename eT>
inline
std::streamsize
coot_ostream::modify_stream(std::ostream& o, const eT* data, const uword n_elem)
  {
  o.unsetf(ios::showbase);
  o.unsetf(ios::uppercase);
  o.unsetf(ios::showpos);

  o.fill(' ');

  std::streamsize cell_width;

  bool use_layout_B = false;
  bool use_layout_C = false;
  bool use_layout_D = false;

  for(uword i=0; i<n_elem; ++i)
    {
    const eT val = data[i];

    if(coot_isfinite(val) == false)  { continue; }

    if(
      ( cond_rel< (sizeof(eT) > 4) && (is_same_type<uword,eT>::yes || is_same_type<sword,eT>::yes) >::geq(val, eT(+10000000000)) )
      ||
      ( cond_rel< (sizeof(eT) > 4) &&  is_same_type<sword,eT>::yes                                 >::leq(val, eT(-10000000000)) )
      )
      {
      use_layout_D = true;
      break;
      }

    if(
      ( val >= eT(+100) )
      ||
      //( (is_signed<eT>::value) && (val <= eT(-100)) ) ||
      //( (is_non_integral<eT>::value) && (val > eT(0)) && (val <= eT(+1e-4)) ) ||
      //( (is_non_integral<eT>::value) && (is_signed<eT>::value) && (val < eT(0)) && (val >= eT(-1e-4)) )
        (
        cond_rel< is_signed<eT>::value >::leq(val, eT(-100))
        )
      ||
        (
        cond_rel< is_non_integral<eT>::value >::gt(val,  eT(0))
        &&
        cond_rel< is_non_integral<eT>::value >::leq(val, eT(+1e-4))
        )
      ||
        (
        cond_rel< is_non_integral<eT>::value && is_signed<eT>::value >::lt(val, eT(0))
        &&
        cond_rel< is_non_integral<eT>::value && is_signed<eT>::value >::geq(val, eT(-1e-4))
        )
      )
      {
      use_layout_C = true;
      break;
      }

    if(
      // (val >= eT(+10)) || ( (is_signed<eT>::value) && (val <= eT(-10)) )
      (val >= eT(+10)) || ( cond_rel< is_signed<eT>::value >::leq(val, eT(-10)) )
      )
      {
      use_layout_B = true;
      }
    }

  if(use_layout_D)
    {
    o.setf(ios::scientific);
    o.setf(ios::right);
    o.unsetf(ios::fixed);
    o.precision(4);
    cell_width = 21;
    }
  else
  if(use_layout_C)
    {
    o.setf(ios::scientific);
    o.setf(ios::right);
    o.unsetf(ios::fixed);
    o.precision(4);
    cell_width = 13;
    }
  else
  if(use_layout_B)
    {
    o.unsetf(ios::scientific);
    o.setf(ios::right);
    o.setf(ios::fixed);
    o.precision(4);
    cell_width = 10;
    }
  else
    {
    o.unsetf(ios::scientific);
    o.setf(ios::right);
    o.setf(ios::fixed);
    o.precision(4);
    cell_width = 9;
    }

  return cell_width;
  }



template<typename eT>
inline
void
coot_ostream::print_elem_zero(std::ostream& o, const bool modify)
  {
  typedef typename promote_type<eT, s16>::result promoted_eT;

  if(modify)
    {
    const ios::fmtflags   save_flags     = o.flags();
    const std::streamsize save_precision = o.precision();

    o.unsetf(ios::scientific);
    o.setf(ios::fixed);
    o.precision(0);

    o << promoted_eT(0);

    o.flags(save_flags);
    o.precision(save_precision);
    }
  else
    {
    o << promoted_eT(0);
    }
  }



template<typename eT>
inline
void
coot_ostream::print_elem(std::ostream& o, const eT& x, const bool modify)
  {
  if(x == eT(0))
    {
    coot_ostream::print_elem_zero<eT>(o, modify);
    }
  else
    {
    coot_ostream::raw_print_elem(o, x);
    }
  }



template<typename eT>
inline
void
coot_ostream::raw_print_elem(std::ostream& o, const eT& x)
  {
  if(is_signed<eT>::value)
    {
    typedef typename promote_type<eT, s16>::result promoted_eT;

    if(coot_isfinite(x))
      {
      o << promoted_eT(x);
      }
    else
      {
      o << ( coot_isinf(x) ? ((x <= eT(0)) ? "-inf" : "inf") : "nan" );
      }
    }
  else
    {
    typedef typename promote_type<eT, u16>::result promoted_eT;

    o << promoted_eT(x);
    }
  }



template<typename eT>
coot_cold
inline
void
coot_ostream::print(std::ostream& o, const Mat<eT>& m, const bool modify)
  {
  coot_extra_debug_sigprint();

  const coot_ostream_state stream_state(o);

  const uword m_n_rows = m.n_rows;
  const uword m_n_cols = m.n_cols;

  if(m.is_empty() == false)
    {
    if(m_n_cols > 0)
      {
      // Transfer the matrix to temporary CPU memory for printing.
      eT* tmp_mem = cpu_memory::acquire<eT>(m.n_elem);
      coot_rt_t::copy_from_dev_mem(tmp_mem, m.get_dev_mem(true), m.n_elem, 1, 0, 0, m.n_elem);

      const std::streamsize cell_width = modify ? coot_ostream::modify_stream(o, tmp_mem, m.n_elem) : o.width();

      if(cell_width > 0)
        {
        for(uword row=0; row < m_n_rows; ++row)
          {
          for(uword col=0; col < m_n_cols; ++col)
            {
            // the cell width appears to be reset after each element is printed,
            // hence we need to restore it
            o.width(cell_width);
            const uword index = col * m_n_rows + row;
            coot_ostream::print_elem(o, m[index], modify);
            }

          o << '\n';
          }
        }
      else
        {
        for(uword row=0; row < m_n_rows; ++row)
          {
          for(uword col=0; col < m_n_cols-1; ++col)
            {
            const uword index = col * m_n_rows + row;
            coot_ostream::print_elem(o, m[index], modify);
            o << ' ';
            }

          const uword last_index = (m_n_cols - 1) * m_n_rows + row;
          coot_ostream::print_elem(o, m[last_index], modify);
          o << '\n';
          }
        }

      cpu_memory::release(tmp_mem);
      }
    }
  else
    {
    if(modify)
      {
      o.unsetf(ios::showbase);
      o.unsetf(ios::uppercase); 
      o.unsetf(ios::showpos);
      o.setf(ios::fixed);
      }

    o << "[matrix size: " << m_n_rows << 'x' << m_n_cols << "]\n";
    }

  o.flush();
  stream_state.restore(o);
  }



template<typename eT>
coot_cold
inline
void
coot_ostream::print(std::ostream& o, const subview<eT>& m, const bool modify)
  {
  coot_extra_debug_sigprint();

  const coot_ostream_state stream_state(o);

  const uword m_n_rows = m.n_rows;
  const uword m_n_cols = m.n_cols;

  if(m.is_empty() == false)
    {
    if(m_n_cols > 0)
      {
      // Transfer the matrix to temporary CPU memory for printing.
      eT* tmp_mem = cpu_memory::acquire<eT>(m.n_elem);
      coot_rt_t::copy_from_dev_mem(tmp_mem, m.m.get_dev_mem(true), m.n_rows, m.n_cols, m.aux_row1, m.aux_col1, m.m.n_rows);

      const std::streamsize cell_width = modify ? coot_ostream::modify_stream(o, tmp_mem, m.n_elem) : o.width();

      if(cell_width > 0)
        {
        for(uword row=0; row < m_n_rows; ++row)
          {
          for(uword col=0; col < m_n_cols; ++col)
            {
            // the cell width appears to be reset after each element is printed,
            // hence we need to restore it
            o.width(cell_width);
            const uword index = col * m_n_rows + row;
            coot_ostream::print_elem(o, m[index], modify);
            }

          o << '\n';
          }
        }
      else
        {
        for(uword row=0; row < m_n_rows; ++row)
          {
          for(uword col=0; col < m_n_cols-1; ++col)
            {
            const uword index = col * m_n_rows + row;
            coot_ostream::print_elem(o, m[index], modify);
            o << ' ';
            }

          const uword last_index = (m_n_cols - 1) * m_n_rows + row;
          coot_ostream::print_elem(o, m[last_index], modify);
          o << '\n';
          }
        }

      cpu_memory::release(tmp_mem);
      }
    }
  else
    {
    if(modify)
      {
      o.unsetf(ios::showbase);
      o.unsetf(ios::uppercase); 
      o.unsetf(ios::showpos);
      o.setf(ios::fixed);
      }

    o << "[matrix size: " << m_n_rows << 'x' << m_n_cols << "]\n";
    }

  o.flush();
  stream_state.restore(o);
  }



template<typename eT>
coot_cold
inline
void
coot_ostream::print(std::ostream& o, const Cube<eT>& x, const bool modify)
  {
  coot_extra_debug_sigprint();

  const coot_ostream_state stream_state(o);

  if(x.is_empty() == false)
    {
    for(uword slice = 0; slice < x.n_slices; ++slice)
      {
      const Mat<eT> tmp(x.slice_get_dev_mem(slice, true), x.n_rows, x.n_cols);

      o << "[cube slice: " << slice << "]\n";
      coot_ostream::print(o, tmp, modify);

      if((slice + 1) < x.n_slices) { o << '\n'; }
      }
    }
  else
    {
    if(modify)
      {
      o.unsetf(ios::showbase);
      o.unsetf(ios::uppercase);
      o.unsetf(ios::showpos);
      o.setf(ios::fixed);
      }

    o << "[cube size: " << x.n_rows << 'x' << x.n_cols << 'x' << x.n_slices << "]\n";
    }

  stream_state.restore(o);
  }



coot_cold
inline
void
coot_ostream::print(std::ostream& o, const SizeMat& S)
  {
  coot_extra_debug_sigprint();

  const coot_ostream_state stream_state(o);

  o.unsetf(ios::showbase);
  o.unsetf(ios::uppercase);
  o.unsetf(ios::showpos);

  o.setf(ios::fixed);

  o << S.n_rows << 'x' << S.n_cols;

  stream_state.restore(o);
  }



coot_cold
inline
void
coot_ostream::print(std::ostream& o, const SizeCube& S)
  {
  coot_extra_debug_sigprint();

  const coot_ostream_state stream_state(o);

  o.unsetf(ios::showbase);
  o.unsetf(ios::uppercase);
  o.unsetf(ios::showpos);

  o.setf(ios::fixed);

  o << S.n_rows << 'x' << S.n_cols << 'x' << S.n_slices;

  stream_state.restore(o);
  }
