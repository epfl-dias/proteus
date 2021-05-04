/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef OPERATORS_HPP_
#define OPERATORS_HPP_

#include <olap/util/parallel-context.hpp>
#include <platform/common/common.hpp>
#include <platform/topology/device-types.hpp>

#include "lib/plugins/output/plugins-output.hpp"
#include "llvm/IR/IRBuilder.h"
#include "olap/expressions/expressions.hpp"
#include "olap/plugins/plugins.hpp"
#include "olap/routing/degree-of-parallelism.hpp"

// Fwd declaration
class Plugin;
class OperatorState;

namespace proteus::traits {

/**
 * Trait controlled by MemMoves (broadcast)
 */
enum class HomReplication {
  UNIQUE, /**< Each element exists in (exactly) one stream */
  BRDCST, /**< Each element exists in all streams */
};

/**
 * Trait controlled by Router
 */
enum class HomParallelization {
  SINGLE,   /**< There is only a single stream */
  PARALLEL, /**< Multiple streams running in parallel */
};
}  // namespace proteus::traits

class Operator {
 public:
  Operator() : parent(nullptr) {}
  virtual ~Operator() { LOG(INFO) << "Collapsing operator"; }
  virtual void setParent(Operator *parent) { this->parent = parent; }
  Operator *const getParent() const { return parent; }
  // Overloaded operator used in checks for children of Join op. More complex
  // cases may require different handling
  bool operator==(
      const Operator &i) const { /*if(this != &i) LOG(INFO) << "NOT EQUAL
                                       OPERATORS"<<this<<" vs "<<&i;*/
    return this == &i;
  }

 protected:
  virtual void produce_(ParallelContext *context) = 0;

 public:
  virtual void produce(ParallelContext *context) final {
    //#ifndef NDEBUG
    //    auto * pip = context->getCurrentPipeline();
    //#endif
    produce_(context);
    //    assert(pip == context->getCurrentPipeline());
  }

  /**
   * Consume is not a const method because Nest does need to keep some state
   * info. Context needs to be passed from the consuming to the producing
   * side to kickstart execution once an HT has been built
   */
  virtual void consume(Context *const context,
                       const OperatorState &childState) = 0;

  virtual RecordType getRowType() const = 0;
  //  {
  //    // FIXME: throw an exception for now, but as soon as existing classes
  //    // implementat it, we should mark it function abstract
  //    throw runtime_error("unimplemented");
  //  }
  /* Used by caching service. Aim is finding whether data to be cached has been
   * filtered by some of the children operators of the plan */
  [[nodiscard]] virtual bool isFiltering() const = 0;

  // Traits

  [[nodiscard]] virtual DeviceType getDeviceType() const = 0;
  [[nodiscard]] virtual DegreeOfParallelism getDOP() const = 0;
  [[nodiscard]] virtual DegreeOfParallelism getDOPServers() const = 0;
  [[nodiscard]] virtual proteus::traits::HomReplication getHomReplication()
      const = 0;
  [[nodiscard]] virtual proteus::traits::HomParallelization
  getHomParallelization() const {
    return (getDOP() == 1) ? proteus::traits::HomParallelization::SINGLE
                           : proteus::traits::HomParallelization::PARALLEL;
  }
  [[nodiscard]] virtual bool isPacked() const = 0;

 private:
  Operator *parent;
};

class UnaryOperator : public Operator {
 public:
  UnaryOperator(Operator *const child) : Operator(), child(child) {}
  ~UnaryOperator() override { LOG(INFO) << "Collapsing unary operator"; }

  [[nodiscard]] virtual Operator *getChild() const { return child; }
  void setChild(Operator *child) { this->child = child; }

  [[nodiscard]] DeviceType getDeviceType() const override {
    return getChild()->getDeviceType();
  }

  [[nodiscard]] DegreeOfParallelism getDOP() const override {
    return getChild()->getDOP();
  }
  [[nodiscard]] DegreeOfParallelism getDOPServers() const override {
    return getChild()->getDOPServers();
  }
  [[nodiscard]] bool isPacked() const override {
    return getChild()->isPacked();
  }
  [[nodiscard]] proteus::traits::HomReplication getHomReplication()
      const override {
    return getChild()->getHomReplication();
  }

 private:
  Operator *child;
};

class BinaryOperator : public Operator {
 public:
  BinaryOperator(Operator *leftChild, Operator *rightChild)
      : Operator(), leftChild(leftChild), rightChild(rightChild) {}
  BinaryOperator(Operator *leftChild, Operator *rightChild,
                 Plugin *const leftPlugin, Plugin *const rightPlugin)
      : Operator(), leftChild(leftChild), rightChild(rightChild) {}
  ~BinaryOperator() override { LOG(INFO) << "Collapsing binary operator"; }
  Operator *getLeftChild() const { return leftChild; }
  Operator *getRightChild() const { return rightChild; }
  void setLeftChild(Operator *leftChild) { this->leftChild = leftChild; }
  void setRightChild(Operator *rightChild) { this->rightChild = rightChild; }

  [[nodiscard]] DeviceType getDeviceType() const override {
    auto dev = getLeftChild()->getDeviceType();
    assert(dev == getRightChild()->getDeviceType());
    return dev;
  }

  [[nodiscard]] DegreeOfParallelism getDOP() const override {
    auto dop = getLeftChild()->getDOP();
    assert(dop == getRightChild()->getDOP());
    return dop;
  }

  [[nodiscard]] DegreeOfParallelism getDOPServers() const override {
    auto dop = getLeftChild()->getDOPServers();
    assert(dop == getRightChild()->getDOPServers());
    return dop;
  }

  [[nodiscard]] bool isPacked() const override {
    auto pckd = getLeftChild()->isPacked();
    assert(pckd == getRightChild()->isPacked());
    return pckd;
  }

 protected:
  Operator *leftChild;
  Operator *rightChild;
};

namespace experimental {
template <typename T>
class POperator : public T {
 public:
  using T::T;

  void consume(Context *const context, const OperatorState &childState) final {
    auto ctx = dynamic_cast<ParallelContext *>(context);
    assert(ctx);

    consume(ctx, childState);
  }

  virtual void consume(ParallelContext *context,
                       const OperatorState &childState) = 0;
};

class Operator : public POperator<::Operator> {
  using POperator::POperator;
};
class UnaryOperator : public POperator<::UnaryOperator> {
  using POperator::POperator;
};
class BinaryOperator : public POperator<::BinaryOperator> {
  using POperator::POperator;
};
}  // namespace experimental

#endif /* OPERATORS_HPP_ */
