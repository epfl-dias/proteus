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

#ifndef PROTEUS_ART_HPP
#define PROTEUS_ART_HPP

#include <oltp/index/ART/OptimisticLockCoupling/art_node.hpp>

#include "platform/common/common.hpp"
#include "platform/threadpool/thread.hpp"
#include "platform/topology/affinity_manager.hpp"
#include "string"

namespace art_olc {

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
  bool insert_recursive_opt(ARTNode *node, ARTNode **nodeRef, V const &value,
                            ARTLeaf<K, V> *newLeaf, size_t depth, bool upsert,
                            ARTNode *&parent, uint64_t parentVersion,
                            bool &needRestart);
  ARTLeaf<K, V> *getByKey_opt(ARTNode *node, key_unit_t *key,
                              bool &needRestart);
  void copyPrefix(ARTNode *__restrict source, const key_unit_t *k, size_t len,
                  size_t depth);
  void copyPrefix(ARTNode *__restrict source, ARTNode *__restrict dst,
                  size_t len);

  void convertKey(key_unit_t *__restrict newkey, K &k) {
    if (typeid(K) != typeid(std::string)) return;
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
    convertKey(newkey, key);
  }
  ARTLeaf<K, V> *res = nullptr;
  bool needRestart = true;
  while (needRestart) {
    needRestart = false;
    res = getByKey_opt(root, newkey, needRestart);
  }

  if (res == nullptr) {
    return {};
  }
  return res->getOneValue();
}

template <class K, class V>
ARTLeaf<K, V> *ART<K, V>::getByKey_opt(ARTNode *node, key_unit_t *key,
                                       bool &needRestart) {
  size_t depth = 0;
  if (node == nullptr) {
    return nullptr;
  }

  auto v = node->readLockOrRestart(needRestart);
  ARTNode *parent = nullptr;
  uint64_t parentVersion = 0;
  if (needRestart) return nullptr;

  while (node != nullptr) {
    v = node->readLockOrRestart(needRestart);
    if (parent != nullptr) {
      parent->readUnlockOrRestart(parentVersion, needRestart);
      if (needRestart) return nullptr;
    }

    node->readUnlockOrRestart(v, needRestart);
    if (needRestart) return nullptr;
    if (unlikely(node->isLeaf())) {
      auto *leaf = static_cast<ARTLeaf<K, V> *>(node);

      if (unlikely(*((K *)key) == *((K *)leaf->_key.getData()))) {
        node->readUnlockOrRestart(v, needRestart);
        return leaf;
      } else {
        node->readUnlockOrRestart(v, needRestart);
        return nullptr;
      }
    }
    if (node->prefix_len) {
      if (node->prefix_match(key, depth) != node->prefix_len) {
        node->readUnlockOrRestart(v, needRestart);
        if (needRestart) return nullptr;
        return nullptr;
      }
      depth += node->prefix_len;
    }
    parent = node;
    parentVersion = v;
    node = static_cast<ARTInnerNode *>(node)->getChild(key[depth]);
    if (node == nullptr) {
      node->readUnlockOrRestart(v, needRestart);
      if (needRestart) return nullptr;
    }
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

  uint64_t versionInfo = 0;
  bool res = false;
  bool needRestart = false;
  ARTNode *parent = nullptr;
  while (!res) {
    needRestart = false;
    res = insert_recursive_opt(root, &root, value, newLeaf, 0, false, parent,
                               versionInfo, needRestart);
  }

  return res;
}

template <class K, class V>
bool ART<K, V>::insert_recursive_opt(ARTNode *node, ARTNode **nodeRef,
                                     V const &value, ARTLeaf<K, V> *newLeaf,
                                     size_t depth, bool upsert,
                                     ARTNode *&parent, uint64_t parentVersion,
                                     bool &needRestart) {
  key_unit_t *key = newLeaf->_key.getData();
  auto v = node != nullptr ? node->readLockOrRestart(needRestart) : 0;
  if (needRestart) {
    return false;
  }

  auto parentD = depth - 1;
  const size_t key_len = sizeof(K);
  if (unlikely(node == nullptr)) {
    *nodeRef = newLeaf;
    return true;
  }

  if (node->prefix_len) {
    auto prefix_mismatch = node->prefix_match(key, depth);
    if (prefix_mismatch != node->prefix_len) {
      if (parent != nullptr && !parent->isLeaf()) {
        parent->upgradeToWriteLockOrRestart(parentVersion, needRestart);
        if (needRestart) return false;
      }
      node->upgradeToWriteLockOrRestart(v, needRestart);
      if (needRestart) {
        if (parent != nullptr && !parent->isLeaf()) parent->writeUnlock();
        return false;
      }

      auto *newNode = ARTNode_4::create(prefix_mismatch);
      copyPrefix(node, newNode, prefix_mismatch);

      newNode->insertChild(key[depth + prefix_mismatch], newLeaf);
      newNode->insertChild(node->prefix[prefix_mismatch], node);

      node->prefix_len -= (prefix_mismatch + 1);
      node->prefix.erase(node->prefix.cbegin(),
                         node->prefix.cbegin() + prefix_mismatch + 1);

      auto temp_node = node;
      if (parent != nullptr && !parent->isLeaf())
        static_cast<ARTInnerNode *>(parent)->insertChild(key[parentD], newNode);
      *nodeRef = newNode;

      temp_node->writeUnlock();
      if (parent != nullptr && !parent->isLeaf()) {
        parent->writeUnlock();
      }
      return true;
    }
    depth += node->prefix_len;
  }

  ARTNode *child =
      node->isLeaf() ? nullptr
                     : static_cast<ARTInnerNode *>(node)->getChild(key[depth]);

  node->checkOrRestart(v, needRestart);
  if (needRestart) return false;

  if (child == nullptr && !node->isLeaf()) {
    auto *newLeafNodePtr = static_cast<ARTNode *>(newLeaf);
    if (node->isFull()) {
      if (parent != nullptr && !parent->isLeaf()) {
        parent->upgradeToWriteLockOrRestart(parentVersion, needRestart);
        if (needRestart) return false;
      }
      node->upgradeToWriteLockOrRestart(v, needRestart);
      if (needRestart) {
        if (parent != nullptr) parent->writeUnlock();
        return false;
      }

      auto *tmp = node;

      node = static_cast<ARTInnerNode *>(node)->grow();
      static_cast<ARTInnerNode *>(node)->insertChild(key[depth],
                                                     newLeafNodePtr);

      if (parent != nullptr && !parent->isLeaf())
        static_cast<ARTInnerNode *>(parent)->insertChild(key[parentD], node);
      *nodeRef = node;

      tmp->writeUnlockObsolete();
      if (parent != nullptr && !parent->isLeaf()) {
        parent->writeUnlock();
      }

    } else {
      // TODO: LOCK current node
      node->upgradeToWriteLockOrRestart(v, needRestart);
      if (needRestart) return false;

      if (parent != nullptr && !parent->isLeaf()) {
        parent->readUnlockOrRestart(parentVersion, needRestart);
        if (needRestart) {
          node->writeUnlock();
          return false;
        }
      }

      static_cast<ARTInnerNode *>(node)->insertChild(key[depth],
                                                     newLeafNodePtr);
      node->writeUnlock();
    }
    return true;
  }

  if (parent != nullptr && !parent->isLeaf()) {
    parent->readUnlockOrRestart(parentVersion, needRestart);
  }
  if (needRestart) return false;

  if (node->isLeaf()) {
    auto tmp_depth = parentD + 1;
    auto *leaf = static_cast<ARTLeaf<K, V> *>(node);

    key_unit_t *key2 = leaf->_key.getData();
    auto key2_len = leaf->_key._key_len;

    size_t newPrefixLength = 0;

    while (key2[tmp_depth + newPrefixLength] ==
               key[tmp_depth + newPrefixLength] &&
           tmp_depth + newPrefixLength < sizeof(K)) {
      newPrefixLength++;

      if (tmp_depth + newPrefixLength == key2_len && key2_len == key_len) {
        break;
      }
    }
    if (tmp_depth + newPrefixLength >= sizeof(K)) return true;
    if (parent != nullptr && !parent->isLeaf())
      parent->upgradeToWriteLockOrRestart(parentVersion, needRestart);
    if (needRestart) return false;
    node->upgradeToWriteLockOrRestart(v, needRestart);
    if (needRestart) {
      if (parent != nullptr && !parent->isLeaf()) parent->writeUnlock();
      return false;
    }

    auto *newNode = static_cast<ARTNode *>(ARTNode_4::create(newPrefixLength));

    copyPrefix(newNode, key, newPrefixLength, tmp_depth);

    static_cast<ARTInnerNode *>(newNode)->insertChild(
        key2[tmp_depth + newPrefixLength], node);

    static_cast<ARTInnerNode *>(newNode)->insertChild(
        key[tmp_depth + newPrefixLength], newLeaf);

    auto temp_node = node;

    //    LOG(INFO) << "split leaf";
    if (parent != nullptr && !parent->isLeaf())
      static_cast<ARTInnerNode *>(parent)->insertChild(key[parentD], newNode);
    else
      *nodeRef = newNode;

    temp_node->writeUnlock();

    if (parent != nullptr && !parent->isLeaf()) parent->writeUnlock();
    return true;
  }

  auto res = insert_recursive_opt(child, &child, value, newLeaf, depth + 1,
                                  upsert, node, v, needRestart);

  return res;
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

}  // namespace art_olc

#endif  // PROTEUS_ART_HPP
