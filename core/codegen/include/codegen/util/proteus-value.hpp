/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#ifndef PROTEUS_PROTEUS_VALUE_HPP
#define PROTEUS_PROTEUS_VALUE_HPP

#include <utility>

namespace llvm {
// forward declaration to avoid including the whole header
class AllocaInst;
class Value;
}  // namespace llvm

class ProteusBareValue {
 protected:
  using value_t = llvm::Value *;

 public:
  value_t value;

  [[deprecated]] ProteusBareValue() = default;
  constexpr ProteusBareValue(value_t value) : value(value) {}
};

class ProteusBareValueMemory {
 protected:
  // Memory location carrying a value (alloca, GEP, shared/global pointer, ...).
  // Historically this was restricted to AllocaInst*, but deferred GPU decode
  // paths may bind directly to non-alloca memory pointers.
  using value_t = llvm::Value *;

 public:
  value_t mem;

  [[deprecated]] ProteusBareValueMemory() = default;
  constexpr ProteusBareValueMemory(value_t mem) : mem(mem) {}
};

template <typename T>
class Nullable : public T {
 public:
  llvm::Value *isNull;

  [[deprecated]] Nullable() = default;
  constexpr Nullable(typename T::value_t v, llvm::Value *isNull)
      : T(std::move(v)), isNull(isNull) {}
};

/**
 * Wrappers for LLVM Value and Alloca.
 * Maintain information such as whether the corresponding value is 'NULL'
 * LLVM's interpretation of 'NULL' for primitive types is not sufficient
 * (e.g., lvvm_null(int) = 0
 */
using ProteusValueMemory = Nullable<ProteusBareValueMemory>;
using ProteusValue = Nullable<ProteusBareValue>;

#endif  // PROTEUS_PROTEUS_VALUE_HPP
