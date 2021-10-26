/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef GMONOIDS_HPP_
#define GMONOIDS_HPP_

#include <platform/common/unsupported-operation.hpp>

#include "olap/operators/monoids.hpp"
#include "olap/util/context.hpp"

namespace gpu {
class Monoid {
 public:
  virtual ~Monoid() {}

  virtual llvm::Value *create(Context *const context,
                              llvm::Value *val_accumulating,
                              llvm::Value *val_in) = 0;

  virtual void createUpdate(Context *const context,
                            llvm::Value *val_accumulating, llvm::Value *val_in);

  virtual void createUpdateFromBareValue(Context *const context,
                                         llvm::Value *val_accumulating,
                                         llvm::Value *val_in);

  virtual llvm::Value *createUnary(Context *context, llvm::Value *val_in) {
    return val_in;
  }

  virtual void destroyUnary(Context *context, llvm::Value *val_in) {}

  virtual void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) = 0;

  virtual llvm::Value *createWarpAggregateToAll(Context *const context,
                                                llvm::Value *val_in);

  virtual llvm::Value *createWarpAggregateTo0(Context *const context,
                                              llvm::Value *val_in) {
    return createWarpAggregateToAll(context, val_in);
  }

  virtual inline std::string to_string() const = 0;

  virtual const ExpressionType *getOutputType(const ExpressionType *eType);
  virtual llvm::Type *getStorageType(Context *context, llvm::Type *updateType);

  static Monoid *get(::Monoid m);
};

class CollectMonoid : public Monoid {
 public:
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createUpdate(Context *const context, llvm::Value *val_accumulating,
                    llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  llvm::Value *createUnary(Context *context, llvm::Value *val_in) override;

  const ExpressionType *getOutputType(const ExpressionType *eType) override;
  llvm::Type *getStorageType(Context *context, llvm::Type *updateType) override;

  inline std::string to_string() const override { return "collect"; }
};

class MaxMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createUpdate(Context *const context, llvm::Value *val_accumulating,
                    llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  inline std::string to_string() const override { return "max"; }
};

class MinMonoid : public Monoid {
 private:
  llvm::Value *evalCondition(Context *const context,
                             llvm::Value *val_accumulating,
                             llvm::Value *val_in);

 public:
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createUpdate(Context *const context, llvm::Value *val_accumulating,
                    llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  inline std::string to_string() const override { return "min"; }
};

class SumMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  inline std::string to_string() const override { return "sum"; }
};

class LogOrMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  llvm::Value *createWarpAggregateToAll(Context *const context,
                                        llvm::Value *val_in) override;

  inline std::string to_string() const override { return "lor"; }
};

class LogAndMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  llvm::Value *createWarpAggregateToAll(Context *const context,
                                        llvm::Value *val_in) override;

  inline std::string to_string() const override { return "land"; }
};

class BitOrMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *val_accumulating,
                      llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  inline std::string to_string() const override { return "bor"; }
};

class BitAndMonoid : public Monoid {
  llvm::Value *create(Context *const context, llvm::Value *al_accumulating,
                      llvm::Value *val_in) override;

  void createAtomicUpdate(
      Context *const context, llvm::Value *accumulator_ptr, llvm::Value *val_in,
      llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic) override;

  inline std::string to_string() const override { return "band"; }
};

}  // namespace gpu

namespace std {
string to_string(const gpu::Monoid &m);
}
#endif /* GMONOIDS_HPP_ */
