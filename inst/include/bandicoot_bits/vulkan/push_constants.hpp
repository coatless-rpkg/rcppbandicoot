// Copyright 2026 Marcus Edel (http://www.kurg.org)
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



template<typename uword_type>
struct fill_push_t
  {
  uword_type out_offset;
  uword_type n_rows;
  uword_type n_cols;
  uword_type M_n_rows;
  uword_type n_slices;
  uword_type M_n_elem_slice;
  uword_type val_offset;
  };



template<typename uword_type>
struct copy_push_t
  {
  uword_type out_offset;
  uword_type out_n_rows;
  uword_type out_n_cols;
  uword_type out_M_n_rows;
  uword_type in_offset;
  uword_type in_n_rows;
  uword_type in_n_cols;
  uword_type in_M_n_rows;
  uword_type n_slices;
  uword_type out_M_n_elem_slice;
  uword_type in_M_n_elem_slice;
  };



template<typename uword_type>
struct accu_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_elem;
  };

template<typename uword_type>
struct accu_subview_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_rows;
  uword_type n_cols;
  uword_type M_n_rows;
  };



template<typename uword_type>
struct dot_push_t
  {
  uword_type out_offset;
  uword_type in1_offset;
  uword_type in2_offset;
  uword_type n_elem;
  };



template<typename uword_type>
struct rel_scalar_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_elem;
  uword_type val_offset;
  };



template<typename uword_type>
struct all_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_rows;
  uword_type n_cols;
  uword_type val_offset;
  };

template<typename uword_type>
struct all_vec_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_elem;
  uword_type val_offset;
  };



template<typename uword_type, typename eT>
struct copy_replace_push_t
  {
  uword_type out_offset;
  uword_type out_n_rows;
  uword_type out_n_cols;
  uword_type out_M_n_rows;
  uword_type in_offset;
  uword_type in_n_rows;
  uword_type in_n_cols;
  uword_type in_M_n_rows;
  uword_type n_slices;
  uword_type out_M_n_elem_slice;
  uword_type in_M_n_elem_slice;
  eT old_val;
  eT new_val;
  };



template<typename uword_type>
struct reduce_dim_push_t
  {
  uword_type dest_offset;
  uword_type src_offset;
  uword_type n_rows;
  uword_type n_cols;
  uword_type dest_mem_incr;
  uword_type src_M_n_rows;
  };



template<typename uword_type>
struct eye_push_t
  {
  uword_type out_offset;
  uword_type n_rows;
  uword_type n_cols;
  };



template<typename uword_type>
struct trans_push_t
  {
  uword_type dest_offset;
  uword_type src_offset;
  uword_type n_rows;
  uword_type n_cols;
  };



template<typename uword_type>
struct reorder_cols_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type order_offset;
  uword_type n_rows;
  uword_type out_n_cols;
  };



template<typename uword_type>
struct index_reduce_push_t
  {
  uword_type dest_offset;
  uword_type src_offset;
  uword_type n_rows;
  uword_type n_cols;
  uword_type dest_mem_incr;
  uword_type src_M_n_rows;
  };



template<typename uword_type>
struct index_vec_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type n_elem;
  uword_type aux_offset;
  };



template<typename uword_type>
struct broadcast_push_t
  {
  uword_type dest_offset;
  uword_type dest_in_offset;
  uword_type src_offset;
  uword_type src_n_rows;
  uword_type src_n_cols;
  uword_type copies_per_row;
  uword_type copies_per_col;
  uword_type dest_M_n_rows;
  uword_type dest_in_M_n_rows;
  uword_type src_M_n_rows;
  };



template<typename uword_type>
struct gather_push_t
  {
  uword_type out_offset;
  uword_type in_offset;
  uword_type idx_offset;
  uword_type n_elem;
  };



template<typename uword_type>
struct gemm_push_t
  {
  uword_type c_offset;
  uword_type c_row_offset;
  uword_type c_col_offset;
  uword_type c_M_n_rows;
  uword_type a_offset;
  uword_type a_row_offset;
  uword_type a_col_offset;
  uword_type a_M_n_rows;
  uword_type b_offset;
  uword_type b_row_offset;
  uword_type b_col_offset;
  uword_type b_M_n_rows;
  uword_type c_n_rows;
  uword_type c_n_cols;
  uword_type K;
  uword_type scalars_offset;
  };



template<typename T>
struct is_dev_mem : std::false_type {};

template<typename eT>
struct is_dev_mem< dev_mem_t<eT> > : std::true_type {};

template<typename eT>
struct is_dev_mem< const dev_mem_t<eT> > : std::true_type {};



template<typename uword_type, typename T>
inline
void
fill_push_val(uint8_t* buf, size_t& off, const T& val)
  {
  if constexpr (is_dev_mem<T>::value)
    {
    off = (off + sizeof(uword_type) - 1) & ~(sizeof(uword_type) - 1);
    uword_type uval = static_cast<uword_type>(val.vk_mem_ptr.offset);
    std::memcpy(buf + off, &uval, sizeof(uword_type));
    off += sizeof(uword_type);
    }
  else if constexpr (std::is_same<T, uword>::value || std::is_same<T, sword>::value)
    {
    off = (off + sizeof(uword_type) - 1) & ~(sizeof(uword_type) - 1);
    uword_type uval = static_cast<uword_type>(val);
    std::memcpy(buf + off, &uval, sizeof(uword_type));
    off += sizeof(uword_type);
    }
  else
    {
    off = (off + sizeof(T) - 1) & ~(sizeof(T) - 1);
    std::memcpy(buf + off, &val, sizeof(T));
    off += sizeof(T);
    }
  }



template<typename uword_type, size_t I, typename... Args>
inline
typename
enable_if2
  <
  (I >= sizeof...(Args)),
  void
  >::result
fill_push_tuple(uint8_t*, size_t&, const std::tuple<Args...>&)
  { }

template<typename uword_type, size_t I, typename... Args>
inline
typename
enable_if2
  <
  (I < sizeof...(Args)),
  void
  >::result
fill_push_tuple(uint8_t* buf, size_t& off, const std::tuple<Args...>& t)
  {
  fill_push_val<uword_type>(buf, off, std::get<I>(t));
  fill_push_tuple<uword_type, I + 1>(buf, off, t);
  }



template<typename uword_type>
inline
void
fill_push_proxies(uint8_t*, size_t&)
  { }

template<typename uword_type, typename T1, typename... Ts>
inline
void
fill_push_proxies(uint8_t* buf, size_t& off, const Proxy<T1>& proxy, const Ts&... rest)
  {
  fill_push_tuple<uword_type, 0>(buf, off, proxy.args());
  fill_push_proxies<uword_type>(buf, off, rest...);
  }



template<size_t I, typename... Args>
inline
typename
enable_if2
  <
  (I >= sizeof...(Args)),
  void
  >::result
collect_dev_mem_tuple(std::vector<VkDescriptorBufferInfo>&, const std::tuple<Args...>&)
  { }

template<size_t I, typename... Args>
inline
typename
enable_if2
  <
  (I < sizeof...(Args)),
  void
  >::result
collect_dev_mem_tuple(std::vector<VkDescriptorBufferInfo>& infos, const std::tuple<Args...>& t)
  {
  using plain_t = std::remove_cv_t<std::remove_reference_t<typename std::tuple_element<I, std::tuple<Args...>>::type>>;
  if constexpr (is_dev_mem<plain_t>::value)
    {
    const auto& mem = std::get<I>(t);
    VkDescriptorBufferInfo info{};
    info.buffer = mem.vk_mem_ptr.buffer;
    info.offset = 0;
    info.range  = VK_WHOLE_SIZE;
    infos.push_back(info);
    }
  collect_dev_mem_tuple<I + 1>(infos, t);
  }



inline
void
collect_dev_mem_infos(std::vector<VkDescriptorBufferInfo>&)
  { }

template<typename T1, typename... Ts>
inline
void
collect_dev_mem_infos(std::vector<VkDescriptorBufferInfo>& infos, const Proxy<T1>& proxy, const Ts&... rest)
  {
  collect_dev_mem_tuple<0>(infos, proxy.args());
  collect_dev_mem_infos(infos, rest...);
  }



template<typename uword_type, typename T>
inline
constexpr
size_t
push_val_size()
  {
  if constexpr (is_dev_mem<T>::value)
    return sizeof(uword_type);
  else if constexpr (std::is_same<T, uword>::value || std::is_same<T, sword>::value)
    return sizeof(uword_type);
  else
    return sizeof(T);
  }



template<typename uword_type, size_t I, typename... Args>
inline
constexpr
typename
enable_if2
  <
  (I >= sizeof...(Args)),
  size_t
  >::result
push_tuple_size_impl()
  {
  return 0;
  }

template<typename uword_type, size_t I, typename... Args>
inline
constexpr
typename
enable_if2
  <
  (I < sizeof...(Args)),
  size_t
  >::result
push_tuple_size_impl()
  {
  return push_val_size<uword_type, typename std::tuple_element<I, std::tuple<Args...>>::type>()
       + push_tuple_size_impl<uword_type, I + 1, Args...>();
  }

template<typename uword_type, typename Tuple, size_t... Is>
inline
constexpr
size_t
push_tuple_size_impl(std::index_sequence<Is...>)
  {
  return (push_val_size<uword_type, typename std::tuple_element<Is, Tuple>::type>() + ...);
  }

template<typename uword_type, typename Tuple>
inline
constexpr
size_t
push_tuple_size()
  {
  return push_tuple_size_impl<uword_type, Tuple>(std::make_index_sequence<std::tuple_size<Tuple>::value>{});
  }

template<typename uword_type>
inline
constexpr
size_t
push_proxies_size()
  {
  return 0;
  }

template<typename uword_type, typename T1, typename... Ts>
inline
constexpr
size_t
push_proxies_size()
  {
  typedef typename std::remove_const<typename std::remove_reference<decltype(std::declval<Proxy<T1>>().args())>::type>::type tuple_t;
  return push_tuple_size<uword_type, tuple_t>() + push_proxies_size<uword_type, Ts...>();
  }


template<typename uword_type, typename T>
inline
size_t
push_val_aligned_next_off(size_t off)
  {
  using PlainT = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (is_dev_mem<PlainT>::value || std::is_same<PlainT, uword>::value || std::is_same<PlainT, sword>::value)
    {
    off = (off + sizeof(uword_type) - 1) & ~(sizeof(uword_type) - 1);
    return off + sizeof(uword_type);
    }
  else
    {
    off = (off + sizeof(PlainT) - 1) & ~(sizeof(PlainT) - 1);
    return off + sizeof(PlainT);
    }
  }



template<typename uword_type, typename Tuple, size_t I>
inline
size_t
push_aligned_tuple_size_from(size_t off)
  {
  if constexpr (I >= std::tuple_size<Tuple>::value)
    return off;
  else
    return push_aligned_tuple_size_from<uword_type, Tuple, I + 1>(
      push_val_aligned_next_off<uword_type, std::tuple_element_t<I, Tuple>>(off));
  }



template<typename uword_type>
inline
size_t
push_aligned_proxies_size_acc(size_t off)
  {
  return off;
  }

template<typename uword_type, typename T1, typename... Ts>
inline
size_t
push_aligned_proxies_size_acc(size_t off)
  {
  typedef typename std::remove_const<typename std::remove_reference<decltype(std::declval<Proxy<T1>>().args())>::type>::type tuple_t;
  return push_aligned_proxies_size_acc<uword_type, Ts...>(
    push_aligned_tuple_size_from<uword_type, tuple_t, 0>(off));
  }



template<typename uword_type, typename... Ts>
inline
size_t
push_aligned_proxies_size()
  {
  return push_aligned_proxies_size_acc<uword_type, Ts...>(0);
  }
