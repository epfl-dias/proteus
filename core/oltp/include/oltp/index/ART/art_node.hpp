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
#ifndef PROTEUS_ART_NODE_HPP
#define PROTEUS_ART_NODE_HPP

#include <array>
#include <bit>
#include <cassert>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <utility>
#include <vector>

#include "oltp/index/ART/art-allocator.hpp"
#include "oltp/index/ART/art_key.hpp"

namespace indexes::art {

static const uint8_t EMPTY = 0;

typedef uint8_t key_unit_t;
typedef size_t len_t;

const std::string tree_print_specifier = "\t";

class ARTNode {
 public:
  friend class ARTNode_4;
  friend class ARTNode_16;
  friend class ARTNode_48;
  friend class ARTNode_256;

  // length of compressed path
  uint16_t prefix_len;
  // valid number of children
  uint8_t n_children;
  //  change deque to vector
  std::vector<uint8_t> prefix;
  // lock for synchronization
  std::atomic<uint64_t> typeVersionLockObsolete{0b100};

  explicit ARTNode(size_t prefix_len) : prefix_len(prefix_len), n_children(0) {}
  virtual ~ARTNode() = default;

  [[nodiscard]] virtual bool isLeaf() const = 0;
  [[nodiscard]] virtual bool isFull() const = 0;
  size_t prefix_match(const uint8_t *key, size_t depth) const;

  static void destroy(ARTNode *node) {
    // TODO
  }

  static bool isLocked(uint64_t version) { return ((version & 0b10) == 0b10); }

  void writeLockOrRestart(bool &needRestart) {
    uint64_t version;
    version = readLockOrRestart(needRestart);
    if (needRestart) {
      return;
    }
    upgradeToWriteLockOrRestart(version, needRestart);
    if (needRestart) return;
  }
  void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
    if (typeVersionLockObsolete.compare_exchange_strong(version,
                                                        version + 0b10)) {
      version = version + 0b10;
    } else {
      needRestart = true;
    }
  }
  void writeUnlock() { typeVersionLockObsolete.fetch_add(0b10); }
  uint64_t readLockOrRestart(bool &needRestart) const {
    uint64_t version;
    version = typeVersionLockObsolete.load();
    if (isLocked(version) || isObsolete(version)) {
      needRestart = true;
    }
    return version;
  }
  void checkOrRestart(uint64_t startRead, bool &needRestart) const {
    readUnlockOrRestart(startRead, needRestart);
  }
  void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
    needRestart = (startRead != typeVersionLockObsolete.load());
  }
  static bool isObsolete(uint64_t version) { return (version & 1) == 1; }
  void writeUnlockObsolete() { typeVersionLockObsolete.fetch_add(0b11); }
};

class ARTInnerNode : public ARTNode {
 public:
  explicit ARTInnerNode(size_t prefix_len) : ARTNode(prefix_len) {}

  [[nodiscard]] inline bool isLeaf() const override { return false; }

  virtual ARTNode *getChild(key_unit_t &partial_key) = 0;
  virtual void insertChild(key_unit_t &partial_key, ARTNode *child) = 0;
  virtual ARTNode *grow() = 0;
  virtual ARTNode *shrink() = 0;
};

template <class K, class V>
class ARTLeafSingleValue;

template <class K, class V>
class ARTLeaf : public ARTNode {
 public:
  ARTKey<K> _key;
  ~ARTLeaf() override = default;

  [[nodiscard]] inline bool isLeaf() const override { return true; }
  [[nodiscard]] inline bool isFull() const override { return true; }

  static ARTLeaf<K, V> *create(ARTKey<K> key, V val, bool single_value) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTLeafSingleValue<K, V>)>::getInstance();

    single_value = true;  // FIXME!!
    if (likely(single_value)) {
      auto *memAlloc = artAllocator.allocate();

      auto *mem = new (memAlloc) ARTLeafSingleValue<K, V>(std::move(key), val);
      return mem;

    } else {
      assert(false);
      return nullptr;
    }
  }

  /* create ARTKey together */
  static ARTLeaf<K, V> *create(const key_unit_t *key, V val,
                               bool single_value) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTLeafSingleValue<K, V>)>::getInstance();
    if (likely(single_value)) {
      auto *memAlloc = artAllocator.allocate();
      auto *mem = new (memAlloc) ARTLeafSingleValue<K, V>(key, val);
      return mem;

    } else {
      assert(false);
      return nullptr;
    }
  }

  virtual V getValByIdx(size_t idx);

  virtual uint count();
  virtual bool empty();
  auto getKey() { return _key; }

  virtual void insert(V const &val);
  virtual void remove(V &val);

  virtual V getOneValue();

 protected:
  explicit ARTLeaf(const key_unit_t *key, V value, size_t const key_len) {
    _key = ARTKey<K>(key_len);
  }

  explicit ARTLeaf(const ARTKey<K> key) : ARTNode(0), _key(std::move(key)) {}
};

/* class for single value leaf nodes */
template <class K, class V>
class ARTLeafSingleValue : public ARTLeaf<K, V> {
 public:
  ~ARTLeafSingleValue() override = default;

  [[nodiscard]] inline bool isLeaf() const override { return true; }
  [[nodiscard]] inline bool isFull() const override { return true; }

  V getValByIdx(size_t) override { return _value; }

  uint count() override { return 1; }
  bool empty() override { return false; }

  void insert(V const &val) override { _value = val; }
  void remove(V &val) override { assert(false); }

  V getOneValue() override { return _value; }

 private:
  V _value;

  explicit ARTLeafSingleValue(ARTKey<K> key, V val)
      : ARTLeaf<K, V>(key), _value(val) {}

  friend class ARTLeaf<K, V>;
};

/* class for multi value leaf nodes */
template <class K, class V>
class ARTLeafMultiValue : public ARTLeaf<K, V> {
 public:
  ~ARTLeafMultiValue() override = default;

  [[nodiscard]] bool isLeaf() const override { return true; }
  [[nodiscard]] bool isFull() const override { return true; }

  V getValByIdx(size_t idx) override {
    if (idx >= _values.size()) {
      throw std::runtime_error("idx is out of the size of the values array");
    }
    return _values[idx];
  }

  auto count() override { return _values.size(); }
  auto empty() override { return _values.empty(); }

  void insert(V const &val) override { _values.emplace_back(val); }
  void remove(V &val) override {
    _values.erase(std::remove(_values.begin(), _values.end(), val),
                  _values.end());
  }

  V getOneValue() override {
    assert(_values.size() >= 1);

    return _values[0];
  }

 private:
  std::vector<V> _values;

  ARTLeafMultiValue(ARTKey<K> key, V val) : ARTLeaf<K, V>(key, val) {
    _values.emplace_back(val);
  }

  friend class ARTLeaf<K, V>;
};

class ARTNode_4 : public ARTInnerNode {
 public:
  explicit ARTNode_4(size_t prefix_len) : ARTInnerNode(prefix_len) {
    _children.fill(nullptr);
    _partial_keys.fill(EMPTY);
  }
  ~ARTNode_4() override = default;

  ARTNode *getChild(key_unit_t &partial_key) override;
  void insertChild(key_unit_t &partial_key, ARTNode *child) override;
  ARTNode *grow() override;
  ARTNode *shrink() override;
  [[nodiscard]] bool isFull() const override;

  static ARTNode_4 *create(size_t prefix_len) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTNode_4)>::getInstance();
    auto *mem = artAllocator.allocate();
    return new (mem) ARTNode_4(prefix_len);
  }

 private:
  std::array<uint8_t, 4> _partial_keys{};
  std::array<ARTNode *, 4> _children{};

  friend class ARTNode_16;
};

class ARTNode_16 : public ARTInnerNode {
 public:
  explicit ARTNode_16(size_t prefix_len) : ARTInnerNode(prefix_len) {
    _children.fill(nullptr);
    for (uint8_t i = 0; i < 16; i++) {
      _partial_keys[i] = i;
    }
  }
  ~ARTNode_16() override = default;

  static ARTNode_16 *create(const ARTNode_4 &node4) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTNode_16)>::getInstance();
    auto *mem = artAllocator.allocate();
    auto *node16 = new (mem) ARTNode_16(node4.prefix_len);

    node16->n_children = 4;
    node16->prefix = node4.prefix;

    std::copy_n(node4._partial_keys.begin(), node4.n_children,
                node16->_partial_keys.begin());
    std::copy_n(node4._children.begin(), node4.n_children,
                node16->_children.begin());
    return node16;
  }

  ARTNode *getChild(key_unit_t &partial_key) override;
  void insertChild(key_unit_t &partial_key, ARTNode *child) override;
  ARTNode *grow() override;
  ARTNode *shrink() override;
  [[nodiscard]] bool isFull() const override;

 private:
  std::array<uint8_t, 16> _partial_keys{};
  std::array<ARTNode *, 16> _children{};

  friend class ARTNode_4;
  friend class ARTNode_48;
};

class ARTNode_48 : public ARTInnerNode {
 public:
  explicit ARTNode_48(size_t prefix_len) : ARTInnerNode(prefix_len) {
    _children.fill(nullptr);
    _indexes.fill(256);
  }

  ~ARTNode_48() override = default;

  static ARTNode_48 *create(const ARTNode_16 &node16) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTNode_48)>::getInstance();
    auto *mem = artAllocator.allocate();
    auto *node48 = new (mem) ARTNode_48(node16.prefix_len);

    node48->n_children = node16.n_children;
    node48->prefix = node16.prefix;

    std::copy_n(node16._children.begin(), node16.n_children,
                node48->_children.begin());
    for (auto i = 0; i < node48->n_children; i++) {
      node48->_indexes[node16._partial_keys[i]] = i;
    }
    return node48;
  }

  ARTNode *getChild(key_unit_t &partial_key) override;
  void insertChild(key_unit_t &partial_key, ARTNode *child) override;
  ARTNode *grow() override;
  ARTNode *shrink() override;
  [[nodiscard]] bool isFull() const override;

 public:
  std::array<uint16_t, 256> _indexes{};
  std::array<ARTNode *, 48> _children{};

  friend class ARTNode_16;
};

class ARTNode_256 : public ARTInnerNode {
 public:
  explicit ARTNode_256(size_t prefix_len) : ARTInnerNode(prefix_len) {
    this->_children.fill(nullptr);
  }

  ~ARTNode_256() override = default;

  static ARTNode_256 *create(const ARTNode_48 &node48) {
    static thread_local auto &artAllocator =
        ARTAllocator<sizeof(ARTNode_256)>::getInstance();
    auto *mem = artAllocator.allocate();
    auto *node256 = new (mem) ARTNode_256(node48.prefix_len);
    node256->n_children = node48.n_children;
    node256->prefix = node48.prefix;

    uint16_t index;
    for (uint partial_key = 0; partial_key <= 255; partial_key++) {
      index = node48._indexes[partial_key];
      if (node48._indexes[partial_key] == 256)
        node256->_children[partial_key] = nullptr;
      else
        node256->_children[partial_key] = node48._children[index];
    }
    return node256;
  }

  ARTNode *getChild(key_unit_t &partial_key) override;
  void insertChild(key_unit_t &partial_key, ARTNode *child) override;
  ARTNode *grow() override;
  ARTNode *shrink() override;
  [[nodiscard]] bool isFull() const override;

 private:
  std::array<ARTNode *, 256> _children{};
  friend class ARTNode_48;
};

}  // namespace indexes::art
#endif  // PROTEUS_ART_NODE_HPP
