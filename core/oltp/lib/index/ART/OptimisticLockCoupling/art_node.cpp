/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#include "oltp/index/ART/OptimisticLockCoupling/art_node.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <platform/topology/topology.hpp>
#include <platform/util/percentile.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

#include "oltp/common/common.hpp"
#include "oltp/common/constants.hpp"
#include "oltp/interface/bench.hpp"

namespace art_olc {

template class ARTKeyFixedWidth<8>;
template class ARTKeyFixedWidth<16>;
template class ARTKeyFixedWidth<32>;
template class ARTKeyFixedWidth<64>;

template class ARTAllocator<4>;
template class ARTAllocator<16>;
template class ARTAllocator<48>;
template class ARTAllocator<256>;

size_t ARTNode::prefix_match(const uint8_t *key, size_t depth) const {
  size_t len;
  for (len = 0; len < this->prefix_len; len++) {
    if (key[depth + len] != this->prefix[len]) {
      return len;
    }
  }
  return len;
}

ARTNode::~ARTNode() {}

ARTNode *ARTNode_4::getChild(key_unit_t &partial_key) {
  for (uint8_t i = 0; i < 4; i++) {
    if (_partial_keys[i] == partial_key) {
      return _children[i];
    }
  }
  return nullptr;
}

void ARTNode_4::insertChild(key_unit_t &partial_key, ARTNode *child) {
  int pos = 0;
  while (pos < this->n_children && partial_key > _partial_keys[pos]) {
    pos++;
  }
  assert(pos < 4 && "Position beyond node capacity");
  if (unlikely(_partial_keys[pos] == partial_key) &&
      _children[pos] != nullptr) {
    // partial key already in the NODE4.
    // Replace child only.
    _children[pos] = child;
  } else {
    if (pos != this->n_children) {
      assert(!this->isFull());
      std::memmove(_partial_keys.data() + pos + 1, _partial_keys.data() + pos,
                   (this->n_children - pos) * sizeof(uint8_t));
      std::memmove(_children.data() + pos + 1, _children.data() + pos,
                   (this->n_children - pos) * sizeof(ARTNode *));
    }
    // insert the partial key and child pointer
    _partial_keys[pos] = partial_key;
    _children[pos] = child;
    this->n_children++;
  }
}

ARTNode *ARTNode_4::grow() {
  auto *n = static_cast<ARTNode_4 *>(this);
  auto new_node = static_cast<ARTNode *>(ARTNode_16::create(*n));
  ARTNode::destroy(this);
  return new_node;
}

ARTNode *ARTNode_4::shrink() {
  throw std::runtime_error("Node_4 can not shrink");
}

bool ARTNode_4::isFull() const { return this->n_children == 4; }

ARTNode *ARTNode_16::getChild(key_unit_t &partial_key) {
#if defined(__i386__) || defined(__amd64__)
  int bitfield = _mm_movemask_epi8(_mm_cmpeq_epi8(
                     _mm_set1_epi8(partial_key),
                     _mm_loadu_si128((__m128i *)&_partial_keys[0]))) &
                 ((1 << ARTNode::n_children) - 1);

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

void ARTNode_16::insertChild(key_unit_t &partial_key, ARTNode *child) {
  int pos = 0;
  while (pos < this->n_children && partial_key > _partial_keys[pos]) {
    pos++;
  }
  assert(pos < 16 && "Position beyond node capacity");
  if (unlikely(_partial_keys[pos] == partial_key &&
               _children[pos] != nullptr)) {
    // partial key already in the NODE16.
    // Replace child only.
    _children[pos] = child;
  } else {
    if (pos != this->n_children) {
      // somewhere in between, move siblings to right.
      assert(!this->isFull());
      std::memmove(_partial_keys.data() + pos + 1, _partial_keys.data() + pos,
                   (this->n_children - pos) * sizeof(uint8_t));
      std::memmove(_children.data() + pos + 1, _children.data() + pos,
                   (this->n_children - pos) * sizeof(ARTNode *));
    }
    // insert the partial key and child pointer
    _partial_keys[pos] = partial_key;
    _children[pos] = child;
    this->n_children++;
  }
}

ARTNode *ARTNode_16::grow() {
  auto *n = static_cast<ARTNode_16 *>(this);
  auto new_node = static_cast<ARTNode *>(ARTNode_48::create(*n));
  return new_node;
}

ARTNode *ARTNode_16::shrink() { return nullptr; }

bool ARTNode_16::isFull() const { return this->n_children == 16; }

ARTNode *ARTNode_48::getChild(key_unit_t &partial_key) {
  uint16_t index = this->_indexes[partial_key];
  return index == 256 ? nullptr : _children[index];
}

void ARTNode_48::insertChild(key_unit_t &partial_key, ARTNode *child) {
  if (n_children < 48 && _indexes[partial_key] == 256) {
    // else add index and add in the end.
    assert(this->n_children < 48 && "Node full.");
    _indexes[partial_key] = this->n_children;
    _children[this->n_children] = child;
    this->n_children++;
  } else {
    // if already there, then update child-pointer directly.
    _children[_indexes[partial_key]] = child;
  }
}

ARTNode *ARTNode_48::grow() {
  auto *n = static_cast<ARTNode_48 *>(this);
  auto new_node = static_cast<ARTNode *>(ARTNode_256::create(*n));
  ARTNode::destroy(this);

  //  delete this;
  return new_node;
}

ARTNode *ARTNode_48::shrink() { return nullptr; }

bool ARTNode_48::isFull() const { return this->n_children == 48; }

ARTNode *ARTNode_256::getChild(key_unit_t &partial_key) {
  return _children[partial_key];
}

void ARTNode_256::insertChild(key_unit_t &partial_key, ARTNode *child) {
  _children[partial_key] = child;
}

ARTNode *ARTNode_256::grow() {
  throw std::runtime_error("Node 256 cannot grow");
}

ARTNode *ARTNode_256::shrink() { return nullptr; }

bool ARTNode_256::isFull() const { return false; }

}  // namespace art_olc
