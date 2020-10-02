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

#include <memory>
#include <mutex>

#include "common/common.hpp"
#include "indexes/ART/adaptive_radix_tree_index.hpp"
#include "indexes/ART/adaptive_radix_tree_nodes.hpp"

namespace indexes {

template void* AdaptiveRadixTreeIndex<uint64_t, void*>::find(uint64_t key);
template void AdaptiveRadixTreeIndex<uint64_t, void*>::insert(uint64_t key,
                                                              void*& value);
template bool AdaptiveRadixTreeIndex<uint64_t, void*>::find(
    BinaryComparableKey<uint64_t> key, void*& val);

template <class K, class V>
bool AdaptiveRadixTreeIndex<K, V>::find(BinaryComparableKey<K> key, V& val) {
  std::shared_ptr<ARTNode<K, V>> curr = _root;
  // std::shared_ptr<ARTNode<K, V>> child = nullptr;

  int depth = 0;
  size_t key_len = key.key_length();
  // LOG(INFO) << "root-cap: "<< (std::dynamic_pointer_cast<ARTInnerNode<K,
  // V>>(_root))->node_capacity();
  LOG(INFO) << "Key: " << key.getRawValue();
  std::bitset<64> kk(key.getRawValue());
  LOG(INFO) << "Actual bits: " << kk;

  while (curr != nullptr) {
    if (curr->isLeaf()) {
      /*
          if leafMatches(node, key, depth)
            return node
          return NULL
       */

      auto leaf = std::dynamic_pointer_cast<ARTLeafNode<K, V>>(curr);
      LOG(INFO) << "isLeaf().:  k:" << leaf->_key;
      if (leaf->_key == key) {
        LOG(INFO) << "returning it";
        val = leaf->_val;
        return true;
      } else {
        return false;
      }
      //
    }
    /*
     if checkPrefix(node,key,depth)!=node.prefixLen
        return NULL
     * */

    auto p = curr->check_prefix(key, depth);
    if (p != curr->_prefix_len) {
      LOG(INFO) << "prefix-mismatch";
      /* prefix mismatch */
      return false;
    }

    /*
        depth=depth+node.prefixLen
        next=findChild(node, key[depth])
        return search(next, key, depth+1)
     */

    LOG(INFO) << "node-cap: "
              << (std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr))
    ->node_capacity();
    depth += curr->_prefix_len;
    std::bitset<8> y(key[depth]);
    LOG(INFO) << "Y: " << y;
    auto next = (std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr))
    ->find_child(key[depth]);
    depth += 1;
    curr = next;
    LOG(INFO) << "inc depth: " << depth;

    //
    //    if (cur->_prefix_len !=
    //        cur->check_prefix(key_arr,key_len,depth )) {
    //      /* prefix mismatch */
    //      return nullptr;
    //    }
    //    if (cur->_prefix_len == key_len - depth) {
    //      /* exact match */
    //      return cur->isLeaf()
    //                 ? &(std::dynamic_pointer_cast<ARTLeafNode<K,
    //                 V>>(cur)->_val) : nullptr;
    //    }
    //    child = std::dynamic_pointer_cast<ARTInnerNode<K,
    //    V>>(cur)->find_child(
    //        key_arr[depth + cur->_prefix_len]);
    //    depth += (cur->_prefix_len + 1);
    //    cur = child != nullptr ? child : nullptr;
  }
  return false;
}

template <class K, class V>
V AdaptiveRadixTreeIndex<K, V>::find(K key) {
  // FIXME: how can it be sure that there will be a value existing!!
  V tmp;
  find(BinaryComparableKey<K>{key}, tmp);
  return tmp;
}

std::mutex st_lock;
//
// template <class K, class V>
// static inline char* loadKey(std::shared_ptr<ARTNode<K, V>> in){
//  return reinterpret_cast<char*>(&(std::dynamic_pointer_cast<ARTLeafNode<K,
//  V>>(in)->_key));
//
//}

template <class K, class V>
inline void AdaptiveRadixTreeIndex<K, V>::replace(
    std::shared_ptr<ARTNode<K, V>> parent,
std::shared_ptr<ARTNode<K, V>> old_node,
std::shared_ptr<ARTNode<K, V>> new_node) {
if (parent == nullptr) {
_root = new_node;
} else {
assert(parent->isLeaf() == false);
(std::dynamic_pointer_cast<ARTInnerNode<K, V>>(parent))
->set_child(old_node->_prefix[0], new_node);
}
}

template <class K, class V>
void AdaptiveRadixTreeIndex<K, V>::insert(K key_in, V& value) {
  std::unique_lock<std::mutex> lk(st_lock);
  BinaryComparableKey<K> key(key_in);
  size_t key_len = key.key_length();

  LOG(INFO) << " --------------------------------------------------------";
  LOG(INFO) << this->name << " - Insert Key: " << key_in;

  LOG(INFO) << "[ART] KeyLen: " << key.key_length();
  std::bitset<64> kb(key_in);
  std::bitset<8> kb2(key[0]);
  LOG(INFO) << "[ART] Key: " << kb << " |  " << kb2;

  if (_root == nullptr) {
    // empty-tree
    LOG(INFO) << "[ART] Case-1";
    _root = std::make_shared<ARTLeafNode<K, V>>(key, value);
    _root->_prefix = key;
    _root->_prefix_len = std::max((uint64_t)1, key_len);
    LOG(INFO) << "[ART][Loop] CurrPrefixLen: " << _root->_prefix_len;
    return;
  }
  //  else{
  //    LOG(INFO) << "===============";
  //    LOG(INFO) << *(this->_root.get());
  //    LOG(INFO) << "===============";
  //  }

  std::shared_ptr<ARTNode<K, V>> curr = _root;
  std::shared_ptr<ARTNode<K, V>> prev = nullptr;
  std::shared_ptr<ARTNode<K, V>> child = nullptr;
  std::shared_ptr<ARTInnerNode<K, V>> curr_inner;
  K partial_key;
  bool is_prefix_match;
  uint depth = 0, prefix_match_len;

  while (true) {
    LOG(INFO) << "[ART][Loop] Depth: " << depth;
    LOG(INFO) << "[ART][Loop] CurrPrefixLen: " << curr->_prefix_len;

    if (curr->isLeaf()) {
      // expand-node
      LOG(INFO) << "[ART][Loop] curr->isLeaf()";
      /* Algo:
          5 newNode=makeNode4()
          6 key2=loadKey(node)
          7 for (i=depth; key[i]==key2[i]; i=i+1)
          8     newNode.prefix[i-depth]=key[i]
          9 newNode.prefixLen=i-depth
          10 depth=depth+newNode.prefixLen
          11 addChild(newNode, key[depth], leaf)
          12 addChild(newNode, key2[depth], node)
          13 replace(node, newNode)
          14 return
       */

      auto newNode = std::make_shared<ARTInnerNode_4<K, V>>();
      auto key2 = (std::dynamic_pointer_cast<ARTLeafNode<K, V>>(curr))->_key;
      auto max = std::min(key2.key_length(), key.key_length());
      uint32_t i;
      for (i = depth; key[i] == key2[i] && i <= max; i++) {
        newNode->_prefix[i - depth] = key[i];
      }

      newNode->_prefix_len = i - depth;
      // LOG(INFO) << "NewNode PrefixLen: " << newNode->_prefix_len << " | i: "
      // << i;
      depth = depth + newNode->_prefix_len;

      // TODO: maybe set other stuff of new_leaf (_prefix / _prefix_len)
      auto new_leaf = std::make_shared<ARTLeafNode<K, V>>(key, value);

      newNode->set_child(key[depth], new_leaf);
      newNode->set_child(key2[depth], curr);
      replace(prev, curr, newNode);

      return;
    }

    auto p = curr->check_prefix(key, depth);
    if (key_in == 256) {
      LOG(INFO) << "depth: " << depth;
      LOG(INFO) << "p: " << p;
      LOG(INFO) << "curr->_prefix_len: " << curr->_prefix_len;
    }
    if (p != curr->_prefix_len) {
      // prefix-mismatch
      LOG(INFO) << "[ART][Loop] prefix-mismatch";

      /* Algo:
          17 newNode=makeNode4()
          18 addChild(newNode, key[depth+p], leaf)
          19 addChild(newNode, node.prefix[p], node)
          20 newNode.prefixLen=p
          21 memcpy(newNode.prefix, node.prefix, p)
          22 node.prefixLen=node.prefixLen-(p+1)
          23 memmove(node.prefix,node.prefix+p+1,node.prefixLen)
          24 replace(node, newNode)
          25 return
       */

      auto newNode = std::make_shared<ARTInnerNode_4<K, V>>();

      // TODO: maybe set other stuff of new_leaf (_prefix / _prefix_len)
      auto new_leaf = std::make_shared<ARTLeafNode<K, V>>(key, value);

      newNode->set_child(key[depth + p], new_leaf);
      newNode->set_child(curr->_prefix[p], curr);

      newNode->_prefix_len = p;
      memcpy(newNode->_prefix.data(), curr->_prefix.data(), p);

      curr->_prefix_len -= (p + 1);
      memmove(curr->_prefix.data(), (curr->_prefix.data()) + p + 1,
              curr->_prefix_len);
      replace(prev, curr, newNode);
      return;
    }

    curr_inner = std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr);

    depth = depth + curr->_prefix_len;

    auto next = curr_inner->find_child(key[depth]);
    if (next == nullptr) {
      std::bitset<8> ss(key[depth]);
      LOG(INFO) << "Add to inner node: " << ss;
      // add to inner node
      if (curr_inner->isFull()) {
        auto grown_up = curr_inner->grow();
        replace(prev, curr, grown_up);
        curr_inner = grown_up;
      }
      auto new_leaf = std::make_shared<ARTLeafNode<K, V>>(key, value);
      // TODO: maybe set other stuff of new_leaf (_prefix, _prefix_len)
      //      std::bitset<8> y(key_byte_array[depth]);
      //      LOG(INFO) << "setting child: " << y;
      curr_inner->set_child(key[depth], new_leaf);
      return;
    } else {
      // recurse.
      prev = curr;
      curr = next;
      depth += 1;
    }
  }

  //    if(curr->isLeaf()){
  //      auto newNode = std::make_shared<ARTInnerNode_4<K, V>>();
  //      auto currLeaf = std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr);
  //
  //
  //      size_t i = 0;
  //      char* node_key = loadKey(curr); //    key2=loadKey(node)
  //      char* newNodePrefix = newNode->getPrefixCharArray();
  //      for(i = depth; key_arr[i] == node_key[i] ; i++){ //for (i=depth;
  //      key[i]==key2[i]; i=i+1)
  //        assert(i <key_len);
  //        newNodePrefix[i-depth] = key_arr[i];
  //        //newNode.prefix[i-depth]=key[i]
  //      }
  //      newNode->_prefix_len = i -depth; //newNode.prefixLen=i-depth
  //
  //      //    depth=depth+newNode.prefixLen
  //      depth = depth + newNode->_prefix_len;
  //
  //
  //  //    addChild(newNode, key[depth], leaf)
  //
  //  //    addChild(newNode, key2[depth], node)
  //        newNode->set_child(node_key[depth],curr);
  //  //    replace(node, newNode)
  //        curr.
  //  //    return
  //        throw std::runtime_error("implement");
  //
  //      // set_child(const uint8_t partial_key,
  //      //                         std::shared_ptr<ARTNode<K, V>> child)
  //
  //
  //      /*
  //       auto new_node = std::make_shared<ARTLeafNode<K, V>>(key, value);
  //      std::copy(key_arr + depth + curr->_prefix_len + 1, key_arr + key_len,
  //                new_node->getPrefixCharArray());
  //      new_node->_prefix_len = key_len - depth - curr->_prefix_len - 1;
  //      curr_inner->set_child(child_partial_key, new_node);
  //      LOG(INFO) << "[ART][Loop] InsertLeaf:";
  //       * */
  //    }
  //
  //    //p=checkPrefix(node, key, depth)
  //    prefix_match_len = curr->check_prefix(key_arr + depth);
  //
  //    if (prefix_match_len !=curr->_prefix_len){ // prefix mismatch
  //      //newNode=makeNode4()
  //
  //      auto newNode = std::make_shared<ARTInnerNode_4<K, V>>();
  //
  //      // void set_child(const uint8_t partial_key,
  //      //                 std::shared_ptr<ARTNode<K, V>> child)
  //
  //      addChild(newNode, key[depth+p], leaf)
  //      addChild(newNode, node.prefix[p], node)
  //      newNode.prefixLen=p
  //      memcpy(newNode.prefix, node.prefix, p)
  //      node.prefixLen=node.prefixLen-(p+1)
  //      memmove(node.prefix,node.prefix+p+1,node.prefixLen)
  //      replace(node, newNode)
  //      return true;
  //
  //
  //      // create new-parent.
  //
  //      auto curr_prefix_arr = curr->getPrefixCharArray();
  //
  //      std::copy(curr_prefix_arr, curr_prefix_arr + prefix_match_len,
  //                new_parent->getPrefixCharArray());
  //
  //      new_parent->_prefix_len = prefix_match_len;
  //      new_parent->set_child(curr_prefix_arr[prefix_match_len], curr);
  //
  //
  //      K old_prefix_val = curr->_prefix;
  //
  //      auto old_prefix = reinterpret_cast<char*>(old_prefix_val);
  //      auto old_prefix_len = curr->_prefix_len;
  //
  //      curr->_prefix_len = old_prefix_len - prefix_match_len - 1;
  //      std::copy(old_prefix + prefix_match_len + 1, old_prefix +
  //      old_prefix_len,
  //                curr_prefix_arr);
  //
  //      auto new_node = std::make_shared<ARTLeafNode<K, V>>(key, value);
  //
  //      std::copy(key_arr + depth + prefix_match_len + 1, key_arr + key_len,
  //                new_node->getPrefixCharArray());
  //
  //      new_node->_prefix_len = key_len - depth - prefix_match_len - 1;
  //      new_parent->set_child(key_arr[depth + prefix_match_len], new_node);
  //
  //      curr = new_parent;
  //      return true;
  //    }
  //
  //    curr_inner = std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr);
  //
  //    depth = depth + curr->_prefix_len;
  //
  //    //next=findChild(node, key[depth])
  //    auto child_partial_key = key_arr[depth];
  //    auto next = curr_inner->find_child(child_partial_key);
  //
  //
  //
  //    if (next == nullptr){
  //      // add to inner node
  //      if (curr_inner->isFull()){
  //        curr_inner->grow();
  //      }
  //
  //      auto new_node = std::make_shared<ARTLeafNode<K, V>>(key, value);
  //      std::copy(key_arr + depth + curr->_prefix_len + 1, key_arr + key_len,
  //                new_node->getPrefixCharArray());
  //      new_node->_prefix_len = key_len - depth - curr->_prefix_len - 1;
  //      curr_inner->set_child(child_partial_key, new_node);
  //      LOG(INFO) << "[ART][Loop] InsertLeaf:";
  //      return true;
  //    } else{
  //      //insert(next, key, leaf, depth+1); // recurse.
  //
  //  //    depth += curr->_prefix_len + 1;
  //       curr = next;
  //    }
  //
  //
  //
  //  while (true) {
  //    LOG(INFO) << "[ART][Loop] Depth:" << depth;
  //    /* number of bytes of the current node's prefix that match the key */
  //    prefix_match_len = curr->check_prefix(key_arr + depth);
  //    //(**cur).check_prefix(key + depth, key_len - depth);
  //
  //    LOG(INFO) << "[ART][Loop] PrefixMatchLen:" << prefix_match_len;
  //
  //    LOG(INFO) << "[ART][Loop] curr->_prefix_len:" << curr->_prefix_len;
  //    LOG(INFO) << "[ART][Loop] :::" << (key_len - depth);
  //    /* true if the current node's prefix matches with a part of the key */
  //    is_prefix_match = (std::min<size_t>(curr->_prefix_len, key_len - depth))
  //    ==
  //                      prefix_match_len;
  //
  //    if (is_prefix_match) {
  //      LOG(INFO) << "[ART][Loop] is_prefix_match";
  //    } else {
  //      LOG(INFO) << "[ART][Loop] NOTTT is_prefix_match";
  //    }
  //
  //    if (is_prefix_match && curr->_prefix_len == key_len - depth) {
  //      LOG(INFO) << "[ART][Loop] Exact Match:";
  //      /* exact match:
  //       * => "replace"
  //       * => replace value of current node.
  //       * => return old value to caller to handle.
  //       *        _                             _
  //       *        |                             |
  //       *       (aa)                          (aa)
  //       *    a /    \ b     +[aaaaa,v3]    a /    \ b
  //       *     /      \      ==========>     /      \
//       * *(aa)->v1  ()->v2             *(aa)->v3  ()->v2
  //       *
  //       */
  //
  //      /* cur must be a leaf */
  //      auto cur_leaf = std::dynamic_pointer_cast<ARTLeafNode<K, V>>(curr);
  //      cur_leaf->_val = value;
  //      //      T *old_value = cur_leaf->value_;
  //      //      cur_leaf->value_ = value;
  //      //      return old_value;
  //    }
  //
  //    if (!is_prefix_match) {
  //      LOG(INFO) << "[ART][Loop] Prefix_MisMatch:";
  //      /* prefix mismatch:
  //       * => new parent node with common prefix and no associated value.
  //       * => new node with value to insert.
  //       * => current and new node become children of new parent node.
  //       *
  //       *        |                        |
  //       *      *(aa)                    +(a)->Ø
  //       *    a /    \ b     +[ab,v3]  a /   \ b
  //       *     /      \      =======>   /     \
//       *  (aa)->v1  ()->v2          *()->Ø +()->v3
  //       *                          a /   \ b
  //       *                           /     \
//       *                        (aa)->v1 ()->v2
  //       *                        /|\      /|\
//       */
  //
  //      // create new-parent.
  //      auto new_parent = std::make_shared<ARTInnerNode_4<K, V>>();
  //      auto curr_prefix_arr = curr->getPrefixCharArray();
  //
  //      std::copy(curr_prefix_arr, curr_prefix_arr + prefix_match_len,
  //                new_parent->getPrefixCharArray());
  //      // new node_4<T>();
  //      //      new_parent->prefix_ = new char[prefix_match_len];
  //      //      std::copy((**cur).prefix_, (**cur).prefix_ + prefix_match_len,
  //      //                new_parent->prefix_);
  //      new_parent->_prefix_len = prefix_match_len;
  //      new_parent->set_child(curr_prefix_arr[prefix_match_len], curr);
  //
  //      // TODO(rafaelkallis): shrink?
  //      /* memmove((**cur).prefix_, (**cur).prefix_ + prefix_match_len + 1, */
  //      /*         (**cur).prefix_len_ - prefix_match_len - 1); */
  //      /* (**cur).prefix_len_ -= prefix_match_len + 1; */
  //
  //      K old_prefix_val = curr->_prefix;
  //
  //      auto old_prefix = reinterpret_cast<char*>(old_prefix_val);
  //      auto old_prefix_len = curr->_prefix_len;
  //
  //      curr->_prefix_len = old_prefix_len - prefix_match_len - 1;
  //      std::copy(old_prefix + prefix_match_len + 1, old_prefix +
  //      old_prefix_len,
  //                curr_prefix_arr);
  //
  //      auto new_node = std::make_shared<ARTLeafNode<K, V>>(key, value);
  //
  //      std::copy(key_arr + depth + prefix_match_len + 1, key_arr + key_len,
  //                new_node->getPrefixCharArray());
  //
  //      new_node->_prefix_len = key_len - depth - prefix_match_len - 1;
  //      new_parent->set_child(key_arr[depth + prefix_match_len], new_node);
  //
  //      curr = new_parent;
  //      return true;
  //    }
  //
  //    /* must be inner node */
  //    curr_inner = std::dynamic_pointer_cast<ARTInnerNode<K, V>>(curr);
  //    auto child_partial_key = key_arr[depth + curr->_prefix_len];
  //    child = curr_inner->find_child(child_partial_key);
  //    LOG(INFO) << "[ART][Loop] MustBeInnerNode:";
  //    if (child == nullptr) {
  //      LOG(INFO) << "[ART][Loop] Child==nullptr:";
  //      /*
  //       * no child associated with the next partial key.
  //       * => create new node with value to insert.
  //       * => new node becomes current node's child.
  //       *
  //       *      *(aa)->Ø              *(aa)->Ø
  //       *    a /        +[aab,v2]  a /    \ b
  //       *     /         ========>   /      \
//       *   (a)->v1               (a)->v1 +()->v2
  //       */
  //
  //      if (curr_inner->isFull()) {
  //        // FIXME: looks shady. maybe grow/shrink should be index func and
  //        take
  //        // nodes as arg.
  //        LOG(INFO) << "[ART][Loop] FullChild:";
  //        curr_inner = curr_inner->grow();
  //      }
  //
  //      auto new_node = std::make_shared<ARTLeafNode<K, V>>(key, value);
  //      // new_node->_prefix = new char[key_len - depth - (**cur).prefix_len_
  //      -
  //      // 1];
  //
  //      std::copy(key_arr + depth + curr->_prefix_len + 1, key_arr + key_len,
  //                new_node->getPrefixCharArray());
  //      new_node->_prefix_len = key_len - depth - curr->_prefix_len - 1;
  //      curr_inner->set_child(child_partial_key, new_node);
  //      // std::dynamic_pointer_cast<ARTInnerNode<K,V>>(curr);
  //      LOG(INFO) << "[ART][Loop] InsertLeaf:";
  //      return true;
  //    }
  //
  //    /* propagate down and repeat:
  //     *
  //     *     *(aa)->Ø                   (aa)->Ø
  //     *   a /    \ b    +[aaba,v3]  a /    \ b     repeat
  //     *    /      \     =========>   /      \     ========>  ...
  //     *  (a)->v1  ()->v2           (a)->v1 *()->v2
  //     */
  //
  //    depth += curr->_prefix_len + 1;
  //    curr = child;
  //    LOG(INFO) << "[ART][Loop] ======================";
  //  }
}

// template <class K, class V>
// bool AdaptiveRadixTreeIndex<K, V>::find(const K &key, V &val) const {
//  throw std::runtime_error("unimplemented");
//}
//
// template <class K, class V>
// bool AdaptiveRadixTreeIndex<K, V>::contains(const K &key) const {
//  throw std::runtime_error("unimplemented");
//}
//
// template <class K, class V>
// bool AdaptiveRadixTreeIndex<K, V>::erase(K key, V &value) {
//  throw std::runtime_error("unimplemented");
//}
//
// std::ostream &operator<<(std::ostream &out,
//                         const indexes::AdaptiveRadixTreeIndex<K, V> &r) {
//  throw std::runtime_error("unimplemented");
//}

// template class AdaptiveRadixTreeIndex<uint64_t, void*>;

}  // namespace indexes

