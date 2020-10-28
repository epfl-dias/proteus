/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef PROTEUS_ADPATIVE_RADIX_TREE_NODES_HPP
#define PROTEUS_ADPATIVE_RADIX_TREE_NODES_HPP

#include <byteswap.h>

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "binary_comparable.hpp"
#include "common/common.hpp"

#if defined(__i386__) || defined(__amd64__)
#include <emmintrin.h>
#endif

// FIXME: how to define EMPTY value??
namespace indexes {

template <class K, class V>
class ARTNode;
template <class K, class V>
class ARTInnerNode;
template <class K, class V>
class ARTInnerNode_4;
template <class K, class V>
class ARTInnerNode_16;
template <class K, class V>
class ARTInnerNode_48;
template <class K, class V>
class ARTInnerNode_256;
template <class K, class V>
class ARTLeafNode;

template <class K, class V>
std::ostream &operator<<(std::ostream &out, const ARTNode<K, V> &r);

const std::string tree_print_specifier = "\t";

template <class K, class V>
class ARTNode {
 public:
  virtual ~ARTNode() = default;
  virtual bool isLeaf() const = 0;

  static constexpr size_t key_type_len = sizeof(K);

  //  inline char *getPrefixCharArray() {
  //
  //    // FIXME: radix is on high bytes or low bytes?
  //    return reinterpret_cast<char *>(&_prefix);
  //
  ////    key.setKeyLen(sizeof(tid));
  ////    reinterpret_cast<uint64_t *>(&key[0])[0] = __builtin_bswap64(tid);
  //  }

  //  static inline uint8_t _getPartialKey(K &in, uint8_t &idx) {
  //    // FIXME: this depends on endianess of the architecture.
  //    // return static_cast<char*>(static_cast<void*>(&in))[idx];
  //    return reinterpret_cast<uint8_t *>(&in)[idx];
  //  }

  size_t check_prefix(BinaryComparableKey<K> &key, uint depth) {
    for (int i = 0; i < std::min(_prefix_len, key.key_length()); i++) {
      if (key.getRawValue() == 256) {
        std::bitset<8> s(_prefix[i]);
        std::bitset<8> t(key[depth + i]);
        LOG(INFO) << "_prefix[i]: " << s;
        LOG(INFO) << "_prefix[i]: " << t;
      }
      if (_prefix[i] != key[depth + i]) return i;
    }
    return _prefix_len;
  }

  virtual void serialize(std::ostream &os, size_t level = 0) const {
    if (this->isLeaf()) {
      auto leafNode = (ARTLeafNode<K, V> *)(this);
      for (auto i = 0; i < level; i++) os << tree_print_specifier;
      os << "L:" << leafNode->_key << std::endl;
    } else {
      auto innerNode = (ARTInnerNode<K, V> *)(this);
      switch (innerNode->node_capacity()) {
        case 4: {
          auto node4 = (ARTInnerNode_4<K, V> *)(this);
          node4->serialize(os, level);
          break;
        }
        case 16: {
          auto node16 = (ARTInnerNode_16<K, V> *)(this);
          node16->serialize(os, level);
          break;
        }
        case 48: {
          auto node48 = (ARTInnerNode_48<K, V> *)(this);
          node48->serialize(os, level);
          break;
        }
        case 256: {
          auto node256 = (ARTInnerNode_256<K, V> *)(this);
          node256->serialize(os, level);
          break;
        }
        default:
          assert(false && "unknown node type");
          break;
      }
    }
  }

  friend std::ostream &operator<<<>(std::ostream &out, const ARTNode<K, V> &r);

 public:
  size_t _prefix_len{};
  BinaryComparableKey<K> _prefix{};
};

template <class K, class V>
class ARTInnerNode : public ARTNode<K, V> {
 public:
  // node-operations
  virtual std::shared_ptr<ARTNode<K, V>> find_child(
      const uint8_t partial_key) = 0;
  virtual void set_child(const uint8_t partial_key,
                         std::shared_ptr<ARTNode<K, V>> child) = 0;
  virtual bool isFull() = 0;
  virtual std::shared_ptr<ARTInnerNode<K, V>> grow() = 0;
  virtual void shrink() { throw std::runtime_error("unimplemented"); }

  // meta-data
  virtual size_t node_capacity() = 0;
  size_t num_children() { return n_children; }
  bool isLeaf() const { return false; }
  ~ARTInnerNode() = default;

 public:
  static constexpr uint8_t EMPTY = 255;
  size_t n_children{};
};

template <class K, class V>
class ARTInnerNode_4 : public ARTInnerNode<K, V> {
 public:
  ARTInnerNode_4() {
    _children.fill(nullptr);
    _partial_keys.fill(ARTInnerNode<K, V>::EMPTY);
  }

  std::shared_ptr<ARTNode<K, V>> find_child(const uint8_t partial_key) {
    for (uint8_t i = 0; i < 4; i++) {
      if (_partial_keys[i] == partial_key) {
        return _children[i];
      }
    }
    return nullptr;
  }
  void set_child(const uint8_t partial_key,
                 std::shared_ptr<ARTNode<K, V>> child) {
    int pos = 0;
    while (pos < this->n_children && partial_key > _partial_keys[pos]) {
      pos++;
    }
    assert(pos < 4 && "Position beyond node capacity");
    if (_partial_keys[pos] == partial_key) {
      // partial key already in the NODE4.
      // Replace child only.
      _children[pos] = child;
      return;
    } else {
      if (pos != this->n_children) {
        // somewhere in between, move siblings to right.
        assert(!this->isFull());
        std::memmove(_partial_keys.data() + pos + 1, _partial_keys.data() + pos,
                     (this->n_children - pos) * sizeof(uint8_t));
        std::memmove(
            _children.data() + pos + 1, _children.data() + pos,
            (this->n_children - pos) * sizeof(std::shared_ptr<ARTNode<K, V>>));
      }
      // insert the partial key and child pointer
      _partial_keys[pos] = partial_key;
      _children[pos] = child;
      this->n_children++;
    }
  }

  void serialize(std::ostream &os, size_t level) const {
    for (auto i = 0; i < this->n_children; i++) {
      for (auto j = 0; j < level; j++) os << tree_print_specifier;
      std::bitset<8> y(_partial_keys[i]);
      os << "K:" << y << std::endl;
      _children[i]->serialize(os, level + 1);
    }
  }

  bool isFull() { return (4 == ARTInnerNode<K, V>::n_children); }
  size_t node_capacity() { return 4; }
  std::shared_ptr<ARTInnerNode<K, V>> grow() {
    return std::make_shared<ARTInnerNode_16<K, V>>(*this);
  }

 private:
  // FIXME: possible optmization: keep keys and children in std::pair to avoid
  // second random access.
  std::array<uint8_t, 4> _partial_keys{};
  std::array<std::shared_ptr<ARTNode<K, V>>, 4> _children{};

  friend class ARTInnerNode_16<K, V>;
};

template <class K, class V>
class ARTInnerNode_16 : public ARTInnerNode<K, V> {
 public:
  ARTInnerNode_16() {
    _children.fill(nullptr);
    _partial_keys.fill(ARTInnerNode<K, V>::EMPTY);
  }
  ARTInnerNode_16(ARTInnerNode_4<K, V> &toGrow) {
    _children.fill(nullptr);
    _partial_keys.fill(ARTInnerNode<K, V>::EMPTY);
    this->_prefix = toGrow._prefix;
    this->_prefix_len = toGrow._prefix_len;
    this->n_children = toGrow.n_children;
    std::copy_n(toGrow._partial_keys.begin(), toGrow.n_children,
                this->_partial_keys.begin());
    std::copy_n(toGrow._children.begin(), toGrow.n_children,
                this->_children.begin());
  }

  std::shared_ptr<ARTNode<K, V>> find_child(const uint8_t partial_key) {
#if defined(__i386__) || defined(__amd64__)
    // FIXME: https://stackoverflow.com/a/11571531
    // m128 has misalignment issue.
    int bitfield = _mm_movemask_epi8(_mm_cmpeq_epi8(
                       _mm_set1_epi8(partial_key),
                       _mm_loadu_si128((__m128i *)&_partial_keys[0]))) &
                   ((1 << ARTInnerNode<K, V>::n_children) - 1);

    return (bool)bitfield ? _children[__builtin_ctz(bitfield)] : nullptr;
#else
    int lo, mid, hi;
    lo = 0;
    hi = n_children;
    while (lo < hi) {
      mid = (lo + hi) / 2;
      if (partial_key < _partial_keys[mid]) {
        hi = mid;
      } else if (partial_key > _partial_keys[mid]) {
        lo = mid + 1;
      } else {
        return _children[mid];
      }
    }
    return nullptr;
#endif
  }

  void set_child(const uint8_t partial_key,
                 std::shared_ptr<ARTNode<K, V>> child) {
    int pos = 0;
    while (pos < this->n_children && partial_key > _partial_keys[pos]) {
      pos++;
    }
    assert(pos < 16 && "Position beyond node capacity");
    if (_partial_keys[pos] == partial_key) {
      // partial key already in the NODE4.
      // Replace child only.
      _children[pos] = child;
      return;
    } else {
      if (pos != this->n_children) {
        // somewhere in between, move siblings to right.
        assert(!this->isFull());
        std::memmove(_partial_keys.data() + pos + 1, _partial_keys.data() + pos,
                     (this->n_children - pos) * sizeof(uint8_t));
        std::memmove(
            _children.data() + pos + 1, _children.data() + pos,
            (this->n_children - pos) * sizeof(std::shared_ptr<ARTNode<K, V>>));
      }
      // insert the partial key and child pointer
      _partial_keys[pos] = partial_key;
      _children[pos] = child;
      this->n_children++;
    }
  }

  void serialize(std::ostream &os, size_t level) const {
    for (auto i = 0; i < this->n_children; i++) {
      for (auto j = 0; j < level; j++) os << tree_print_specifier;
      std::bitset<8> y(_partial_keys[i]);
      os << "K:" << y << std::endl;
      _children[i]->serialize(os, level + 1);
    }
  }

  bool isFull() { return (16 == ARTInnerNode<K, V>::n_children); }
  size_t node_capacity() { return 16; }
  std::shared_ptr<ARTInnerNode<K, V>> grow() {
    return std::make_shared<ARTInnerNode_48<K, V>>(*this);
  }

 private:
  std::array<uint8_t, 16> __attribute__((aligned(16))) _partial_keys{};
  std::array<std::shared_ptr<ARTNode<K, V>>, 16> _children{};

  friend class ARTInnerNode_48<K, V>;
};

template <class K, class V>
class ARTInnerNode_48 : public ARTInnerNode<K, V> {
 public:
  ARTInnerNode_48() {
    _children.fill(nullptr);
    _index_to_child.fill(ARTInnerNode<K, V>::EMPTY);
  }
  ARTInnerNode_48(ARTInnerNode_16<K, V> &toGrow) {
    _children.fill(nullptr);
    _index_to_child.fill(ARTInnerNode<K, V>::EMPTY);
    this->_prefix = toGrow._prefix;
    this->_prefix_len = toGrow._prefix_len;
    this->n_children = toGrow.n_children;
    std::copy_n(toGrow._children.begin(), toGrow.n_children,
                this->_children.begin());
    for (auto i = 0; i < this->n_children; i++) {
      this->_index_to_child[toGrow._partial_keys[i]] = i;
    }
  }

  std::shared_ptr<ARTNode<K, V>> find_child(const uint8_t partial_key) {
    uint8_t index = _index_to_child[partial_key];
    return ARTInnerNode<K, V>::EMPTY != index ? _children[index] : nullptr;

    // return _children[_index_to_child[partial_key]];
  }

  void set_child(const uint8_t partial_key,
                 std::shared_ptr<ARTNode<K, V>> child) {
    if (_index_to_child[partial_key] < 48) {
      // if already there, then update child-pointer directly.
      _children[_index_to_child[partial_key]] = child;
    } else {
      // else add index and add in the end.
      assert(this->n_children < 48 && "Node full.");
      _index_to_child[partial_key] = this->n_children;
      _children[this->n_children] = child;
      this->n_children++;
    }
  }

  void serialize(std::ostream &os, size_t level) const {
    for (auto i = 0; i < 256; i++) {
      if (_index_to_child[i] != ARTInnerNode<K, V>::EMPTY) {
        for (auto j = 0; j < level; j++) os << tree_print_specifier;
        std::bitset<8> y(i);
        os << "K:" << y << std::endl;
        _children[i]->serialize(os, level + 1);
      }
    }
  }

  bool isFull() { return (48 == ARTInnerNode<K, V>::n_children); }
  size_t node_capacity() { return 48; }
  std::shared_ptr<ARTInnerNode<K, V>> grow() {
    return std::make_shared<ARTInnerNode_256<K, V>>(*this);
  }

 private:
  std::array<uint8_t, 256> _index_to_child{};
  std::array<std::shared_ptr<ARTNode<K, V>>, 48> _children{};

  friend class ARTInnerNode_256<K, V>;
};

template <class K, class V>
class ARTInnerNode_256 : public ARTInnerNode<K, V> {
 public:
  ARTInnerNode_256() { _children.fill(nullptr); }
  ARTInnerNode_256(ARTInnerNode_48<K, V> &toGrow) {
    _children.fill(nullptr);
    this->_prefix = toGrow._prefix;
    this->_prefix_len = toGrow._prefix_len;
    this->n_children = toGrow.n_children;
    uint8_t index;
    for (uint partial_key = 0; partial_key <= 255; partial_key++) {
      index = toGrow._index_to_child[partial_key];
      if (index != ARTInnerNode<K, V>::EMPTY) {
        this->set_child(partial_key, toGrow._children[index]);
      }
    }
  }

  std::shared_ptr<ARTNode<K, V>> find_child(const uint8_t partial_key) {
    return _children[partial_key];
  }

  void set_child(const uint8_t partial_key,
                 std::shared_ptr<ARTNode<K, V>> child) {
    _children[partial_key] = child;
  }

  void serialize(std::ostream &os, size_t level) const {
    for (auto i = 0; i < 256; i++) {
      if (_children[i] != nullptr) {
        for (auto j = 0; j < level; j++) os << tree_print_specifier;
        std::bitset<8> y(i);
        os << "K:" << y << std::endl;
        _children[i]->serialize(os, level + 1);
      }
    }
  }

  bool isFull() { return (256 == ARTInnerNode<K, V>::n_children); }
  size_t node_capacity() { return 256; }
  std::shared_ptr<ARTInnerNode<K, V>> grow() {
    throw std::runtime_error("Maximum size node, cannot grow");
  }

 private:
  std::array<std::shared_ptr<ARTNode<K, V>>, 256> _children{};
};

template <class K, class V>
class ARTLeafNode : public ARTNode<K, V> {
 public:
  ARTLeafNode(K key, V val) : _key(key), _val(val) {}
  ARTLeafNode(BinaryComparableKey<K> key, V val)
      : _key(std::move(key)), _val(val) {}

  bool isLeaf() const { return true; }

  const V _val;
  const BinaryComparableKey<K> _key;
};

// Outstream implementations.

template <class K, class V>
std::ostream &operator<<(std::ostream &out, const ARTNode<K, V> &r) {
  out << "\n";
  r.serialize(out);
  return out;
}

}  // namespace indexes

#endif  // PROTEUS_ADPATIVE_RADIX_TREE_NODES_HPP
