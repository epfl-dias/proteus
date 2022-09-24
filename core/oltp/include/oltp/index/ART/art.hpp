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
                                           portions thereof, and that both
notices appear in supporting documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
        DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
            RESULTING FROM THE USE OF THIS SOFTWARE.
                */

#ifndef PROTEUS_ART_HPP
#define PROTEUS_ART_HPP

#include <oltp/index/ART/art_node.hpp>

#include "platform/common/common.hpp"
#include "platform/threadpool/thread.hpp"
#include "platform/topology/affinity_manager.hpp"
#include "string"

namespace art_index {

template <class K, class V>
class ART {
 public:
  /* use mult-value leaf node */
  const bool is_unique;
  bool read_only;

  explicit ART() : is_unique(false), read_only(false), root(nullptr) { init(); }
  explicit ART(bool constraint_unique)
      : is_unique(constraint_unique), read_only(false), root(nullptr) {
    init();
  }

  ~ART() = default;

 private:
  void init() {
    ARTAllocator<sizeof(ARTNode_4)>::getInstance().allocate();
    ARTAllocator<sizeof(ARTNode_16)>::getInstance().allocate();
    ARTAllocator<sizeof(ARTNode_48)>::getInstance().allocate();
    ARTAllocator<sizeof(ARTNode_256)>::getInstance().allocate();
    ARTAllocator<sizeof(ARTLeafSingleValue<K, V>)>::getInstance().allocate();
  }

 public:
  V find(K key);
  void freeze() {}
  bool insert(K &key, V const &value);
  bool remove(K key) { return false; }
  bool remove(K key, V value) { return false; }

 private:
  ARTNode *root;

 private:
  bool insert_recursive(ARTNode *&node, key_unit_t *key, V const &value,
                        ARTLeaf<K, V> *newLeaf, size_t depth, bool upsert);
  ARTLeaf<K, V> *getByKey(ARTNode *node, key_unit_t *__restrict key);

  void copyPrefix(ARTNode *__restrict source, const key_unit_t *k, size_t len,
                  size_t depth);
  void copyPrefix(ARTNode *__restrict source, ARTNode *__restrict dst,
                  size_t len);

  void convertKey(key_unit_t *__restrict newkey, K &k) {
    if (typeid(k).name() != typeid(std::string).name()) return;
    switch (sizeof(K)) {
      case 16:
        *((K *)newkey) = __builtin_bswap16(k);
        break;
      case 32:
        *((K *)newkey) = __builtin_bswap32(k);
        break;
      case 64:
        *((K *)newkey) = __builtin_bswap64(k);
        break;
      default:
        std::reverse(newkey, newkey + sizeof(k));
    }
  }
};

template <class K, class V>
V ART<K, V>::find(K key) {
  auto *newkey = reinterpret_cast<key_unit_t *>(&key);

  if (likely(std::endian::native == std::endian::little)) {
    //    convertKey(newkey, key);
  }

  auto *res = getByKey(root, newkey);
  if (res == nullptr) {
    return {};
  }
  return res->getOneValue();
}

template <class K, class V>
ARTLeaf<K, V> *ART<K, V>::getByKey(ARTNode *node, key_unit_t *key) {
  size_t depth = 0;
  if (unlikely(node == nullptr)) {
    return nullptr;
  }
  while (node != nullptr) {
    if (unlikely(node->isLeaf())) {
      auto *leaf = static_cast<ARTLeaf<K, V> *>(node);

      if (unlikely(*((K *)key) == *((K *)leaf->_key.getData()))) {
        return leaf;
      } else {
        return nullptr;
      }
    }

    if (node->prefix_len) {
      if (node->prefix_match(key, depth) != node->prefix_len) {
        return nullptr;
      }
      depth += node->prefix_len;
    }

    node = static_cast<ARTInnerNode *>(node)->getChild(key[depth]);
    depth++;
  }
  return nullptr;
}

template <class K, class V>
bool ART<K, V>::insert(K &key, V const &value) {
  if (this->read_only) {
    return false;
  }

  auto *newkey = reinterpret_cast<key_unit_t *>(&key);
  if (likely(std::endian::native == std::endian::little)) {
    convertKey(newkey, key);
  }

  ARTKey<K> tmp(newkey, sizeof(K));
  auto *newLeaf = ARTLeaf<K, V>::create(std::move(tmp), value, this->is_unique);

  bool res = insert_recursive(root, newkey, value, newLeaf, 0, false);
  return res;
}

template <class K, class V>
bool ART<K, V>::insert_recursive(ARTNode *&node, key_unit_t *key,
                                 V const &value, ARTLeaf<K, V> *newLeaf,
                                 size_t depth, bool upsert) {
  const size_t key_len = sizeof(K);
  if (unlikely(node == nullptr)) {
    node = newLeaf;
    return true;
  }
  if (likely(node->isLeaf())) {
    auto *leaf = static_cast<ARTLeaf<K, V> *>(node);
    key_unit_t *key2 = leaf->_key.getData();
    auto key2_len = leaf->_key._key_len;
    size_t newPrefixLength = 0;
    while (key2[depth + newPrefixLength] == key[depth + newPrefixLength] &&
           depth + newPrefixLength < sizeof(K)) {
      newPrefixLength++;
      if (depth + newPrefixLength == key2_len && key2_len == key_len) {
        if (unlikely(upsert)) {
          leaf->insert(value);
          return true;
        } else {
          throw std::runtime_error("key already present");
        }
      }
    }

    auto *newNode = static_cast<ARTNode *>(ARTNode_4::create(newPrefixLength));
    copyPrefix(newNode, key, newPrefixLength, depth);
    static_cast<ARTInnerNode *>(newNode)->insertChild(
        key[depth + newPrefixLength], newLeaf);
    static_cast<ARTInnerNode *>(newNode)->insertChild(
        key2[depth + newPrefixLength], node);
    node = newNode;
    return true;
  }

  if (node->prefix_len) {
    auto prefix_mismatch = node->prefix_match(key, depth);
    if (prefix_mismatch != node->prefix_len) {
      auto *newNode = ARTNode_4::create(prefix_mismatch);
      copyPrefix(node, newNode, prefix_mismatch);
      newNode->insertChild(key[depth + prefix_mismatch], newLeaf);
      newNode->insertChild(node->prefix[prefix_mismatch], node);
      node->prefix_len -= (prefix_mismatch + 1);
      node->prefix.erase(node->prefix.cbegin(),
                         node->prefix.cbegin() + prefix_mismatch + 1);
      node = newNode;
      return true;
    }
    depth += node->prefix_len;
  }
  ARTNode *child = static_cast<ARTInnerNode *>(node)->getChild(key[depth]);
  ARTNode *child_cpy = child;
  if (child != nullptr) {
    auto res = insert_recursive(child, key, value, newLeaf, depth + 1, upsert);
    if (child_cpy != child) {
      static_cast<ARTInnerNode *>(node)->insertChild(key[depth], child);
    }
    return res;
  }
  auto *newLeafNodePtr = static_cast<ARTNode *>(newLeaf);
  if (node->isFull()) {
    node = static_cast<ARTInnerNode *>(node)->grow();
    static_cast<ARTInnerNode *>(node)->insertChild(key[depth], newLeafNodePtr);
    return true;
  }
  static_cast<ARTInnerNode *>(node)->insertChild(key[depth], newLeafNodePtr);
  return true;
}

template <class K, class V>
void ART<K, V>::copyPrefix(ARTNode *source, const key_unit_t *k, size_t len,
                           size_t depth) {
  assert(sizeof(K) >= depth + len);
  source->prefix.resize(len);
  for (auto i = 0; i < len; i++) {
    source->prefix[i] = k[depth + i];
  }
}

template <class K, class V>
void ART<K, V>::copyPrefix(ARTNode *source, ARTNode *dst, size_t len) {
  assert(source->prefix_len >= len);
  dst->prefix.resize(len);
  for (auto i = 0; i < len; i++) {
    dst->prefix[i] = source->prefix[i];
  }
}

}  // namespace art_index

#endif  // PROTEUS_ART_HPP
