/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_OPERATOR_STATE_HPP
#define PROTEUS_OPERATOR_STATE_HPP

#include <olap/values/expressionTypes.hpp>

class Operator;

class attribute_not_found_in_state : public std::out_of_range {
 public:
  static std::string createWhatFromBindings(
      const RecordAttribute &key,
      const map<RecordAttribute, ProteusValueMemory> &vars) {
    std::stringstream ss;
    ss << "Looking for: " << key;
    for (const auto &v : vars) {
      ss << "  Active binding: " << v.first;
    }
    return ss.str();
  }

  attribute_not_found_in_state(
      const RecordAttribute &key,
      const map<RecordAttribute, ProteusValueMemory> &vars,
      std::out_of_range &source)
      : std::out_of_range(createWhatFromBindings(key, vars) + '\n' +
                          source.what()) {}
};

class OperatorState {
 public:
  OperatorState(const Operator &producer,
                const map<RecordAttribute, ProteusValueMemory> &vars)
      : producer(producer), activeVariables(vars) {}
  OperatorState(const OperatorState &opState)
      : producer(opState.producer), activeVariables(opState.activeVariables) {
    LOG(INFO) << "[Operator State: ] Copy Constructor";
  }
  OperatorState(const Operator &producer, const OperatorState &opState)
      : OperatorState(producer, opState.activeVariables) {}

  [[deprecated]] const map<RecordAttribute, ProteusValueMemory> &getBindings()
      const {
    return activeVariables;
  }

  [[nodiscard]] const Operator &getProducer() const { return producer; }

  [[nodiscard]] bool contains(const RecordAttribute &key) const {
    return activeVariables.count(key) > 0;
  }

  const ProteusValueMemory &operator[](const RecordAttribute &key) const {
    try {
      return activeVariables.at(key);
    } catch (std::out_of_range &e) {
      for (const auto &v : activeVariables) {
        if (v.first.getAttrName() == key.getAttrName()) return v.second;
      }
      throw attribute_not_found_in_state(key, activeVariables, e);
    }
  }

 private:
  const Operator &producer;
  // Variable bindings produced by operator and provided to its parent
  // const map<string, AllocaInst*>& activeVariables;
  const map<RecordAttribute, ProteusValueMemory> &activeVariables;
};

#endif /* PROTEUS_OPERATOR_STATE_HPP */
