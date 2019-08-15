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

#ifndef EXPRESSIONS_HPP_
#define EXPRESSIONS_HPP_

#include "expressions/binary-operators.hpp"
#include "operators/monoids.hpp"
#include "util/context.hpp"
#include "values/expressionTypes.hpp"

class ExprVisitor;        // Forward declaration
class ExprTandemVisitor;  // Forward declaration

// Careful: Using a namespace to avoid conflicts witfh LLVM namespace
namespace expressions {

class RecordProjection;

enum ExpressionId {
  CONSTANT,
  RAWVALUE,
  ARGUMENT,
  RECORD_PROJECTION,
  RECORD_CONSTRUCTION,
  IF_THEN_ELSE,
  BINARY,
  MERGE,
  EXPRESSION_HASHER,
  TESTNULL_EXPRESSION,
  EXTRACT,
  NEG_EXPRESSION,
  CAST_EXPRESSION
};

class Expression {
 public:
  Expression(const ExpressionType *type) : type(type), registered(false) {}

  template <typename T>
  friend class ExpressionCRTP;

 public:
  virtual ~Expression() = default;

  const ExpressionType *getExpressionType() const { return type; }
  virtual ProteusValue accept(ExprVisitor &v) const = 0;
  virtual ProteusValue acceptTandem(ExprTandemVisitor &v,
                                    const expressions::Expression *) const = 0;
  virtual ExpressionId getTypeID() const = 0;

  virtual bool operator<(const expressions::Expression &r) const final;

  virtual inline bool isRegistered() const { return registered; }

  virtual inline void registerAs(string relName, string attrName) {
    registered = true;
    this->relName = relName;
    this->attrName = attrName;
  }

  virtual inline void registerAs(RecordAttribute *attr) {
    registerAs(attr->getRelationName(), attr->getAttrName());
  }

 protected:
  virtual inline Expression &as_expr(string relName, string attrName) = 0;

  virtual inline Expression &as_expr(RecordAttribute *attr) = 0;

 public:
  virtual inline string getRegisteredAttrName() const {
    if (!registered) {
      string error_msg = string("Expression not registered");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    return attrName;
  }

  virtual inline string getRegisteredRelName() const {
    if (!registered) {
      string error_msg = string("Expression not registered");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    return relName;
  }

  virtual RecordAttribute getRegisteredAs() const {
    if (!registered) {
      string error_msg = string("Expression not registered");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    return RecordAttribute{getRegisteredRelName(), getRegisteredAttrName(),
                           getExpressionType()};
  }

 private:
  const ExpressionType *type;

 protected:
  bool registered;
  string relName;
  string attrName;
};

template <typename T, typename Interface = Expression>
class ExprVisitorVisitable : public Interface {
 protected:
  using Interface::Interface;

 public:
  virtual ProteusValue accept(ExprVisitor &v) const override;
  virtual ProteusValue acceptTandem(
      ExprTandemVisitor &v, const expressions::Expression *expr) const override;

 protected:
  virtual inline Expression &as_expr(string relName, string attrName) override {
    ExprVisitorVisitable<T, Expression>::registerAs(relName, attrName);
    return *this;
  }

  virtual inline Expression &as_expr(RecordAttribute *attr) override {
    ExprVisitorVisitable<T, Expression>::registerAs(attr);
    return *this;
  }

 public:
  virtual inline T &as(string relName, string attrName) {
    return static_cast<T &>(
        static_cast<decltype(*this)>(as_expr(relName, attrName)));
  }

  virtual inline T &as(RecordAttribute *attr) {
    return static_cast<T &>(static_cast<decltype(*this)>(
        as_expr(attr->getRelationName(), attr->getAttrName())));
  }

  friend T;
};

template <typename T>
class ExpressionCRTP : public ExprVisitorVisitable<T, Expression> {
 protected:
  using ExprVisitorVisitable<T, Expression>::ExprVisitorVisitable;

  virtual inline bool operator<(const T &r) const = 0;
};

class RecordProjection;

}  // namespace expressions

class expression_t final : public expressions::ExpressionCRTP<expression_t> {
 public:
  typedef expressions::Expression concept_t;

 private:
  std::shared_ptr<const concept_t> data;

  template <typename T>
  expression_t(std::shared_ptr<T> &&ptr)
      : expressions::ExpressionCRTP<expression_t>(ptr->getExpressionType()),
        data(ptr) {
    if (data->isRegistered()) {
      registerAs(data->getRegisteredRelName(), data->getRegisteredAttrName());
    }
  }

 public:
  [[deprecated]] expression_t(const expressions::Expression *ptr)
      : ExpressionCRTP(ptr->getExpressionType()), data(ptr) {
    // not very correct (=who has the ownership?) but should work for now
    if (data->isRegistered()) {
      registerAs(data->getRegisteredRelName(), data->getRegisteredAttrName());
    }
  }

  template <typename T,
            typename =
                std::enable_if_t<std::is_base_of_v<expressions::Expression, T>>,
            typename = std::enable_if_t<!std::is_same_v<expression_t, T>>>
  expression_t(T v) : expression_t(std::make_shared<T>(v)) {}

  expression_t(bool v);
  expression_t(int32_t v);
  expression_t(int64_t v);
  expression_t(double v);
  expression_t(std::string v);
  expression_t(const char *v);

  inline ProteusValue accept(ExprVisitor &v) const override {
    return data->accept(v);
  }

  inline ProteusValue acceptTandem(ExprTandemVisitor &v,
                                   const expression_t &e) const {
    return data->acceptTandem(v, e.data.get());
  }

  [[deprecated]] inline ProteusValue acceptTandem(
      ExprTandemVisitor &v, const expressions::Expression *e) const override {
    return data->acceptTandem(v, e);
  }

  inline expressions::ExpressionId getTypeID() const override {
    return data->getTypeID();
  }

  template <typename T, typename... Args>
  static expression_t make(Args &&... args) {
    return {std::make_shared<T>(args...)};
  }

  [[deprecated]] const concept_t *getUnderlyingExpression() const {
    /* by definition this function leaks data... */
    (void)(new std::shared_ptr<const concept_t>(data));
    // we cannot safely delete the shared_ptr
    return data.get();
  }

  [[deprecated]] operator const concept_t *() const { return data.get(); }

  expressions::RecordProjection operator[](RecordAttribute proj) const;

  virtual inline bool operator<(const expression_t &r) const override final {
    return *data < *(r.data);
  }
};

// Careful: Using a namespace to avoid conflicts witfh LLVM namespace
namespace expressions {

class Constant : public Expression {
 public:
  enum ConstantType { INT, INT64, BOOL, FLOAT, STRING, DSTRING, DATE };
  Constant(const ExpressionType *type) : Expression(type) {}

  virtual ConstantType getConstantType() const = 0;
};

template <typename T>
class ConstantExpressionCRTP : public ExprVisitorVisitable<T, Constant> {
 protected:
  using ExprVisitorVisitable<T, Constant>::ExprVisitorVisitable;
};

template <typename T, typename Tproteus, Constant::ConstantType TcontType,
          typename Texpr>
class TConstant : public ConstantExpressionCRTP<Texpr> {
 private:
  T val;

 protected:
  TConstant(T val, const ExpressionType *type)
      : ConstantExpressionCRTP<Texpr>(type), val(val) {}

 public:
  TConstant(T val) : TConstant(val, new Tproteus()) {}

  T getVal() const { return val; }

  ExpressionId getTypeID() const { return CONSTANT; }

  Constant::ConstantType getConstantType() const { return TcontType; }

  inline bool operator<(const Texpr &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      if (getConstantType() == r.getConstantType()) {
        return this->getVal() < r.getVal();
      } else {
        return this->getConstantType() < r.getConstantType();
      }
    }
    return this->getTypeID() < r.getTypeID();
  }
};

class IntConstant
    : public TConstant<int, IntType, Constant::ConstantType::INT, IntConstant> {
 public:
  using TConstant<int, IntType, Constant::ConstantType::INT,
                  IntConstant>::TConstant;
};

class StringConstant
    : public TConstant<std::string, StringType, Constant::ConstantType::STRING,
                       StringConstant> {
 public:
  using TConstant<std::string, StringType, Constant::ConstantType::STRING,
                  StringConstant>::TConstant;
};

class Int64Constant
    : public TConstant<int64_t, Int64Type, Constant::ConstantType::INT64,
                       Int64Constant> {
 public:
  using TConstant<int64_t, Int64Type, Constant::ConstantType::INT64,
                  Int64Constant>::TConstant;
};

class DateConstant
    : public TConstant<int64_t, DateType, Constant::ConstantType::DATE,
                       DateConstant> {
  static_assert(sizeof(time_t) == sizeof(int64_t), "expected 64bit time_t");

 public:
  using TConstant<int64_t, DateType, Constant::ConstantType::DATE,
                  DateConstant>::TConstant;
};

class BoolConstant
    : public TConstant<bool, BoolType, Constant::ConstantType::BOOL,
                       BoolConstant> {
 public:
  using TConstant<bool, BoolType, Constant::ConstantType::BOOL,
                  BoolConstant>::TConstant;
};

class FloatConstant
    : public TConstant<double, FloatType, Constant::ConstantType::FLOAT,
                       FloatConstant> {
 public:
  using TConstant<double, FloatType, Constant::ConstantType::FLOAT,
                  FloatConstant>::TConstant;
};

class DStringConstant
    : public TConstant<int, DStringType, Constant::ConstantType::FLOAT,
                       DStringConstant> {
 public:
  DStringConstant(int val, void *dictionary)
      : TConstant(val, new DStringType(dictionary)) {}
  ~DStringConstant() = default;
};

/*
 * Conceptually:  What every next() call over a collection produces
 * In the general case, it is a record.
 * However it can be a primitive (i.e., if iterating over [1,2,3,4,5]
 * or even a random collection - (i.e., if iterating over
 * [1,[5,9],[{"a":1},{"a":2}]]
 *
 * XXX How do we specify the schema of the last expression?
 */
class InputArgument : public ExpressionCRTP<InputArgument> {
 public:
  InputArgument(const RecordType *type, int argNo = 0)
      : ExpressionCRTP(type), argNo(argNo) {}

  [[deprecated]] InputArgument(const ExpressionType *type, int argNo,
                               list<RecordAttribute> projections)
      : ExpressionCRTP(type), argNo(argNo) {  //, projections(projections) {
    assert(dynamic_cast<const RecordType *>(type) && "Expected Record Type");

    // FIXME: we should be able to remove this constructor, but for now it's
    // used for the case that projections contains activeLoop, while the type
    // does not. I believe that this originates from legacy code

    // Due to the above, the following sentence is wrong:
    // projections should be a subpart of type->getProjections()
    //     assert(projections.size() <= getProjections().size() &&
    //            "Type mismatches projections");
    // #ifndef NDEBUG
    //     std::cout << type->getType() << std::endl;
    //     // for (const auto &arg :
    //     //      RecordType(*dynamic_cast<const RecordType
    //     *>(type)).getArgsMap()) {
    //     //   std::cout << arg.first << std::endl;
    //     // }

    //     for (const auto &proj : projections) {
    //       std::cout << proj.getAttrName() << std::endl;
    //       assert((proj.getAttrName() == "activeTuple" ||
    //               dynamic_cast<const RecordType *>(type)->getArg(
    //                   proj.getAttrName())) &&
    //              "Attribute not found in RecordType");
    //     }
    // #endif
  }

  int getArgNo() const { return argNo; }
  // list<RecordAttribute> getProjections() const { return projections; }
  list<RecordAttribute> getProjections() const {
    auto rec = dynamic_cast<const RecordType *>(getExpressionType());
    std::list<RecordAttribute> largs;
    for (const auto &attr : rec->getArgs()) {
      largs.emplace_back(*attr);
    }
    return largs;
  }
  ExpressionId getTypeID() const { return ARGUMENT; }
  inline bool operator<(const InputArgument &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      /* Is it the same record? */
      const ExpressionType *lExpr = this->getExpressionType();
      const ExpressionType *rExpr = r.getExpressionType();
      bool cmpExprType1 = *lExpr < *rExpr;
      bool cmpExprType2 = *rExpr < *rExpr;
      bool eqExprType = !cmpExprType1 && !cmpExprType2;
      /* Does this make sense? Do I need equality? */
      if (eqExprType) {
        list<RecordAttribute> lProj = this->getProjections();
        list<RecordAttribute> rProj = r.getProjections();
        if (lProj.size() != rProj.size()) {
          return lProj.size() < rProj.size();
        }

        list<RecordAttribute>::iterator itLeftArgs = lProj.begin();
        list<RecordAttribute>::iterator itRightArgs = rProj.begin();

        while (itLeftArgs != lProj.end()) {
          RecordAttribute attrLeft = (*itLeftArgs);
          RecordAttribute attrRight = (*itRightArgs);

          bool eqAttr = !(attrLeft < attrRight) && !(attrRight < attrLeft);
          if (!eqAttr) {
            return attrLeft < attrRight;
          }
          itLeftArgs++;
          itRightArgs++;
        }
        return false;
      } else {
        return cmpExprType1;
      }
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

  inline expressions::RecordProjection operator[](RecordAttribute proj) const;

 private:
  /**
   * ArgumentNo is meant to represent e.g. the left or right child of a Join,
   * NOT the projections that we need!
   *
   * argNo = 0 => lhs of Join
   * argNo = 1 => rhs of Join, and so on.
   */
  int argNo;
  /**
   * Used as if 'slicing' a record ->
   * Treated as record fields
   * NOTE: One of them (activeLoop) is virtual
   */
  // list<RecordAttribute> projections;
};

class RecordProjection : public ExpressionCRTP<RecordProjection> {
 public:
  [[deprecated]] RecordProjection(ExpressionType *type, Expression *expr,
                                  RecordAttribute attribute)
      : ExpressionCRTP(type), expr(expr), attribute(attribute) {
    assert(type->getTypeID() == this->attribute.getOriginalType()->getTypeID());
    registerAs(getRelationName(), getProjectionName());
  }
  [[deprecated]] RecordProjection(const ExpressionType *type, Expression *expr,
                                  RecordAttribute attribute)
      : ExpressionCRTP(type), expr(expr), attribute(attribute) {
    assert(type->getTypeID() == this->attribute.getOriginalType()->getTypeID());
    registerAs(getRelationName(), getProjectionName());
  }
  RecordProjection(expression_t expr, RecordAttribute attribute)
      : ExpressionCRTP(attribute.getOriginalType()),
        expr(std::move(expr)),
        attribute(attribute) {
    registerAs(getRelationName(), getProjectionName());
  }
  ~RecordProjection() = default;

  expression_t getExpr() const { return expr; }
  string getOriginalRelationName() const {
    return attribute.getOriginalRelationName();
  }
  string getRelationName() const { return attribute.getRelationName(); }
  string getProjectionName() const { return attribute.getAttrName(); }
  RecordAttribute getAttribute() const { return attribute; }

  ExpressionId getTypeID() const { return RECORD_PROJECTION; }
  inline bool operator<(const RecordProjection &r) const {
    //        if (this->getTypeID() == r.getTypeID()) {
    //            cout << "Record Proj Hashing" << endl;
    //            const RecordProjection& rProj =
    //                    dynamic_cast<const RecordProjection&>(r);
    //            bool cmpAttribute1 = this->getAttribute() <
    //            rProj.getAttribute(); bool cmpAttribute2 =
    //            rProj.getAttribute() < this->getAttribute(); bool eqAttribute
    //            = !cmpAttribute1 && !cmpAttribute2;
    //            /* Does this make sense? Do I need equality? */
    //            if (eqAttribute) {
    //                cout << this->getAttribute().getAttrName() << " vs " <<
    //                rProj.getAttribute().getAttrName() << endl; cout <<
    //                this->getAttribute().getRelationName() << " vs " <<
    //                rProj.getAttribute().getRelationName() << endl;
    //                //return this->getExpr() < rProj.getExpr();
    //                return this->getRelationName() < rProj.getRelationName();
    //            } else {
    //                cout << "No record proj match "<<endl;
    //                cout << this->getAttribute().getAttrName() << " vs "
    //                        << rProj.getAttribute().getAttrName() << endl;
    //                cout << this->getAttribute().getRelationName() << " vs "
    //                << rProj.getAttribute().getRelationName() << endl;
    ////                return cmpAttribute1;
    //                return cmpAttribute1 ? cmpAttribute1 : cmpAttribute2;
    //            }
    //        } else {
    //            return this->getTypeID() < r.getTypeID();
    //        }
    if (this->getTypeID() == r.getTypeID()) {
      string n1 = this->getRelationName();
      string n2 = r.getRelationName();

      bool cmpRel1 = this->getRelationName() < r.getRelationName();
      bool cmpRel2 = r.getRelationName() < this->getRelationName();
      bool eqRelation = !cmpRel1 && !cmpRel2;
      if (eqRelation) {
        bool cmpAttribute1 = this->getAttribute() < r.getAttribute();
        bool cmpAttribute2 = r.getAttribute() < this->getAttribute();
        bool eqAttribute = !cmpAttribute1 && !cmpAttribute2;
        if (eqAttribute) {
          return false;
        } else {
          return cmpAttribute1;
        }
      } else {
        return cmpRel1;
      }

    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr;
  RecordAttribute attribute;
};

class HashExpression : public ExpressionCRTP<HashExpression> {
 public:
  HashExpression(expression_t expr)
      : ExpressionCRTP(new Int64Type()), expr(std::move(expr)) {}

  expression_t getExpr() const { return expr; }
  ExpressionId getTypeID() const { return EXPRESSION_HASHER; }
  inline bool operator<(const HashExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return this->getExpr() < r.getExpr();
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr;
};

class ProteusValueExpression : public ExpressionCRTP<ProteusValueExpression> {
 public:
  ProteusValueExpression(const ExpressionType *type, ProteusValue expr)
      : ExpressionCRTP(type), expr(expr) {}

  ProteusValue getValue() const { return expr; }
  ExpressionId getTypeID() const { return RAWVALUE; }
  inline bool operator<(const ProteusValueExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return (expr.value == nullptr && r.expr.value == nullptr)
                 ? (expr.isNull < r.expr.isNull)
                 : (expr.value < r.expr.value);
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  ProteusValue expr;
};

class AttributeConstruction {
 public:
  AttributeConstruction(string name, expression_t expr)
      : name(name), expr(std::move(expr)) {}
  string getBindingName() const { return name; }
  expression_t getExpression() const { return expr; }
  /* Don't need explicit op. overloading */
 private:
  string name;
  expression_t expr;
};

/*
 * XXX
 * I think that unless it belongs to the final result, it is desugarized!!
 */
class RecordConstruction : public ExpressionCRTP<RecordConstruction> {
 public:
  [[deprecated]] RecordConstruction(const ExpressionType *type,
                                    const list<AttributeConstruction> &atts)
      : ExpressionCRTP(type), atts(atts) {
    assert(type->getTypeID() == RECORD);
  }

  RecordConstruction(const list<AttributeConstruction> &atts)
      : ExpressionCRTP(constructRecordType(atts)), atts(atts) {}

  ExpressionId getTypeID() const { return RECORD_CONSTRUCTION; }
  const list<AttributeConstruction> &getAtts() const { return atts; }
  inline bool operator<(const RecordConstruction &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      list<AttributeConstruction> lAtts = this->getAtts();
      list<AttributeConstruction> rAtts = r.getAtts();

      if (lAtts.size() != rAtts.size()) {
        return lAtts.size() < rAtts.size();
      }
      list<AttributeConstruction>::iterator itLeft = lAtts.begin();
      list<AttributeConstruction>::iterator itRight = rAtts.begin();

      while (itLeft != lAtts.end()) {
        if (itLeft->getExpression() != itRight->getExpression()) {
          return itLeft->getExpression() < itRight->getExpression();
        }
        itLeft++;
        itRight++;
      }
      return false;
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

  RecordType *constructRecordType(const list<AttributeConstruction> &attrs) {
    vector<RecordAttribute *> recs;
    for (const auto &a : attrs) {
      auto *type = a.getExpression().getExpressionType();
      auto attr = new RecordAttribute{"tmp", a.getBindingName(), type};
      recs.emplace_back(attr);
    }
    return new RecordType(recs);
  }

 private:
  const list<AttributeConstruction> atts;
};

class IfThenElse : public ExpressionCRTP<IfThenElse> {
 public:
  [[deprecated]] IfThenElse(const ExpressionType *type, Expression *expr1,
                            Expression *expr2, Expression *expr3)
      : ExpressionCRTP(type), expr1(expr1), expr2(expr2), expr3(expr3) {}
  IfThenElse(expression_t expr1, expression_t expr2, expression_t expr3)
      : ExpressionCRTP(expr2.getExpressionType()),
        expr1(std::move(expr1)),
        expr2(std::move(expr2)),
        expr3(std::move(expr3)) {
    assert(expr2.getExpressionType()->getTypeID() ==
           expr3.getExpressionType()->getTypeID());
  }

  ExpressionId getTypeID() const { return IF_THEN_ELSE; }
  expression_t getIfCond() const { return expr1; }
  expression_t getIfResult() const { return expr2; }
  expression_t getElseResult() const { return expr3; }
  virtual inline bool operator<(const expressions::IfThenElse &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      expression_t lCond = this->getIfCond();
      expression_t lIfResult = this->getIfResult();
      expression_t lElseResult = this->getElseResult();

      expression_t rCond = r.getIfCond();
      expression_t rIfResult = r.getIfResult();
      expression_t rElseResult = r.getElseResult();

      bool eqCond = !(lCond < rCond) && !(rCond < lCond);
      bool eqIfResult = !(lIfResult < rIfResult) && !(rIfResult < lIfResult);
      bool eqElseResult =
          !(lElseResult < rElseResult) && !(rElseResult < lElseResult);

      if (eqCond) {
        if (eqIfResult) {
          if (eqElseResult) {
            return false;
          } else {
            return lElseResult < rElseResult;
          }
        } else {
          return lIfResult < rIfResult;
        }
      } else {
        return lCond < rCond;
      }
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr1;
  expression_t expr2;
  expression_t expr3;
};

class BinaryExpression : public Expression {
 public:
  BinaryExpression(const ExpressionType *type, expressions::BinaryOperator *op,
                   expression_t lhs, expression_t rhs)
      : Expression(type), lhs(std::move(lhs)), rhs(std::move(rhs)), op(op) {}

  virtual expression_t getLeftOperand() const { return lhs; }
  virtual expression_t getRightOperand() const { return rhs; }
  expressions::BinaryOperator *getOp() const { return op; }

  virtual ExpressionId getTypeID() const { return BINARY; }

  // virtual inline bool operator<(const BinaryExpression &r) const {
  //   if (this->getTypeID() == r.getTypeID()) {
  //     const BinaryExpression &rBin = dynamic_cast<const BinaryExpression
  //     &>(r); if (this->getOp()->getID() == rBin.getOp()->getID()) {
  //       string error_msg = string(
  //           "[This abstract bin. operator is NOT responsible for this
  //           case!]");
  //       LOG(ERROR) << error_msg;
  //       throw runtime_error(error_msg);
  //     } else {
  //       return this->getOp()->getID() < rBin.getOp()->getID();
  //     }
  //   } else {
  //     return this->getTypeID() < r.getTypeID();
  //   }
  // }

 private:
  const expression_t lhs;
  const expression_t rhs;
  BinaryOperator *op;
};

template <typename T>
class BinaryExpressionCRTP : public ExprVisitorVisitable<T, BinaryExpression> {
 public:
  using ExprVisitorVisitable<T, BinaryExpression>::ExprVisitorVisitable;

  virtual ~BinaryExpressionCRTP() override = default;
};

template <typename T, typename Top>
class TBinaryExpression : public BinaryExpressionCRTP<T> {
  typedef BinaryExpressionCRTP<T> Tparent;

 protected:
  TBinaryExpression(const ExpressionType *type, expression_t lhs,
                    expression_t rhs)
      : Tparent(type, new Top(), std::move(lhs), std::move(rhs)) {}

 public:
  ExpressionId getTypeID() const { return BINARY; }

  inline bool operator<(const T &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      if (this->getOp()->getID() == r.getOp()->getID()) {
        auto l1 = this->getLeftOperand();
        auto l2 = this->getRightOperand();

        auto r1 = r.getLeftOperand();
        auto r2 = r.getRightOperand();

        bool eq1 = l1 < r1;
        bool eq2 = l2 < r2;

        if (eq1) {
          if (eq2) {
            return false;
          } else {
            return l2 < r2;
          }
        } else {
          return l1 < r1;
        }
      } else {
        return this->getOp()->getID() < r.getOp()->getID();
      }
    }
    return this->getTypeID() < r.getTypeID();
  }
};

class EqExpression : public TBinaryExpression<EqExpression, Eq> {
 public:
  EqExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class NeExpression : public TBinaryExpression<NeExpression, Neq> {
 public:
  NeExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class GeExpression : public TBinaryExpression<GeExpression, Ge> {
 public:
  GeExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class GtExpression : public TBinaryExpression<GtExpression, Gt> {
 public:
  GtExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class LeExpression : public TBinaryExpression<LeExpression, Le> {
 public:
  LeExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class LtExpression : public TBinaryExpression<LtExpression, Lt> {
 public:
  LtExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(new BoolType(), lhs, rhs) {}
};

class AddExpression : public TBinaryExpression<AddExpression, Add> {
 public:
  AddExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class SubExpression : public TBinaryExpression<SubExpression, Sub> {
 public:
  SubExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class MultExpression : public TBinaryExpression<MultExpression, Mult> {
 public:
  MultExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class DivExpression : public TBinaryExpression<DivExpression, Div> {
 public:
  DivExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class ModExpression : public TBinaryExpression<ModExpression, Mod> {
 public:
  ModExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(rhs.getExpressionType(), lhs, rhs) {}
};

class AndExpression : public TBinaryExpression<AndExpression, And> {
 public:
  AndExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class OrExpression : public TBinaryExpression<OrExpression, Or> {
 public:
  OrExpression(expression_t lhs, expression_t rhs)
      : TBinaryExpression(lhs.getExpressionType(), lhs, rhs) {}
};

class MaxExpression : public BinaryExpressionCRTP<MaxExpression> {
 public:
  MaxExpression(expression_t lhs, expression_t rhs)
      : BinaryExpressionCRTP(lhs.getExpressionType(), new Max(), lhs, rhs),
        cond(expression_t::make<GtExpression>(lhs, rhs), lhs, rhs) {}

  const IfThenElse *getCond() const { return &cond; }
  inline bool operator<(const MaxExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      if (this->getOp()->getID() == r.getOp()->getID()) {
        return this->cond < r.cond;
      } else {
        return this->getOp()->getID() < r.getOp()->getID();
      }
    }
    return this->getTypeID() < r.getTypeID();
  }

 private:
  IfThenElse cond;
};

class MinExpression : public BinaryExpressionCRTP<MinExpression> {
 public:
  MinExpression(expression_t lhs, expression_t rhs)
      : BinaryExpressionCRTP(lhs.getExpressionType(), new Min(), lhs, rhs),
        cond(expression_t::make<LtExpression>(lhs, rhs), lhs, rhs) {}

  const IfThenElse *getCond() const { return &cond; }
  inline bool operator<(const MinExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      if (this->getOp()->getID() == r.getOp()->getID()) {
        return this->cond < r.cond;
      } else {
        return this->getOp()->getID() < r.getOp()->getID();
      }
    }
    return this->getTypeID() < r.getTypeID();
  }

 private:
  IfThenElse cond;
};

class NegExpression : public ExpressionCRTP<NegExpression> {
 public:
  NegExpression(expression_t expr)
      : ExpressionCRTP(expr.getExpressionType()), expr(expr) {}

  expression_t getExpr() const { return expr; }
  ExpressionId getTypeID() const { return NEG_EXPRESSION; }
  inline bool operator<(const NegExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return this->getExpr() < r.getExpr();
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr;
};

class TestNullExpression : public ExpressionCRTP<TestNullExpression> {
 public:
  TestNullExpression(expression_t expr, bool nullTest = true)
      : ExpressionCRTP(new BoolType()),
        expr(std::move(expr)),
        nullTest(nullTest) {}

  bool isNullTest() const { return nullTest; }
  expression_t getExpr() const { return expr; }
  ExpressionId getTypeID() const { return TESTNULL_EXPRESSION; }
  inline bool operator<(const TestNullExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return this->getExpr() < r.getExpr();
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr;
  bool nullTest;
};

enum class extract_unit {
  MILLISECOND,
  SECOND,
  MINUTE,
  HOUR,
  DAYOFWEEK,
  ISO_DAYOFWEEK,
  DAYOFMONTH,
  DAYOFYEAR,
  WEEK,
  MONTH,
  QUARTER,
  YEAR,
  ISO_YEAR,
  DECADE,
  CENTURY,
  MILLENNIUM
};

class ExtractExpression : public ExpressionCRTP<ExtractExpression> {
 public:
  ExtractExpression(expression_t expr, extract_unit unit)
      : ExpressionCRTP(createReturnType(unit)),
        expr(std::move(expr)),
        unit(unit) {
    assert(this->expr.getExpressionType()->getTypeID() == DATE);
  }

  extract_unit getExtractUnit() const { return unit; }
  expression_t getExpr() const { return expr; }
  ExpressionId getTypeID() const { return EXTRACT; }
  inline bool operator<(const ExtractExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return this->getExpr() < r.getExpr();
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  static ExpressionType *createReturnType(extract_unit u);

  expression_t expr;
  extract_unit unit;
};

class CastExpression : public ExpressionCRTP<CastExpression> {
 public:
  CastExpression(ExpressionType *cast_to, expression_t expr)
      : ExpressionCRTP(cast_to), expr(std::move(expr)) {}

  ~CastExpression() {}

  expression_t getExpr() const { return expr; }
  ExpressionId getTypeID() const { return CAST_EXPRESSION; }
  inline bool operator<(const CastExpression &r) const {
    if (this->getTypeID() == r.getTypeID()) {
      return this->getExpr() < r.getExpr();
    } else {
      return this->getTypeID() < r.getTypeID();
    }
  }

 private:
  expression_t expr;
};

}  // namespace expressions

//===----------------------------------------------------------------------===//
// "Visitor" responsible for generating the appropriate code per Expression
// 'node'
//===----------------------------------------------------------------------===//
class ExprVisitor {
 public:
  virtual ProteusValue visit(const expressions::IntConstant *e) = 0;
  virtual ProteusValue visit(const expressions::Int64Constant *e) = 0;
  virtual ProteusValue visit(const expressions::DateConstant *e) = 0;
  virtual ProteusValue visit(const expressions::FloatConstant *e) = 0;
  virtual ProteusValue visit(const expressions::BoolConstant *e) = 0;
  virtual ProteusValue visit(const expressions::StringConstant *e) = 0;
  virtual ProteusValue visit(const expressions::DStringConstant *e) = 0;
  virtual ProteusValue visit(const expressions::InputArgument *e) = 0;
  virtual ProteusValue visit(const expressions::RecordProjection *e) = 0;
  virtual ProteusValue visit(const expressions::RecordConstruction *e) = 0;
  virtual ProteusValue visit(const expressions::IfThenElse *e) = 0;
  virtual ProteusValue visit(const expressions::EqExpression *e) = 0;
  virtual ProteusValue visit(const expressions::NeExpression *e) = 0;
  virtual ProteusValue visit(const expressions::GeExpression *e) = 0;
  virtual ProteusValue visit(const expressions::GtExpression *e) = 0;
  virtual ProteusValue visit(const expressions::LeExpression *e) = 0;
  virtual ProteusValue visit(const expressions::LtExpression *e) = 0;
  virtual ProteusValue visit(const expressions::AddExpression *e) = 0;
  virtual ProteusValue visit(const expressions::SubExpression *e) = 0;
  virtual ProteusValue visit(const expressions::MultExpression *e) = 0;
  virtual ProteusValue visit(const expressions::DivExpression *e) = 0;
  virtual ProteusValue visit(const expressions::ModExpression *e) = 0;
  virtual ProteusValue visit(const expressions::AndExpression *e) = 0;
  virtual ProteusValue visit(const expressions::OrExpression *e) = 0;
  virtual ProteusValue visit(const expressions::ProteusValueExpression *e) = 0;
  virtual ProteusValue visit(const expressions::MinExpression *e) = 0;
  virtual ProteusValue visit(const expressions::MaxExpression *e) = 0;
  virtual ProteusValue visit(const expressions::HashExpression *e) = 0;
  virtual ProteusValue visit(const expressions::TestNullExpression *e) = 0;
  virtual ProteusValue visit(const expressions::NegExpression *e) = 0;
  virtual ProteusValue visit(const expressions::ExtractExpression *e) = 0;
  virtual ProteusValue visit(const expressions::CastExpression *e1) = 0;
  //    virtual ProteusValue visit(const expressions::MergeExpression *e)      =
  //    0;
  virtual ~ExprVisitor() {}
};

class ExprTandemVisitor {
 public:
  virtual ProteusValue visit(const expressions::IntConstant *e1,
                             const expressions::IntConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::Int64Constant *e1,
                             const expressions::Int64Constant *e2) = 0;
  virtual ProteusValue visit(const expressions::DateConstant *e1,
                             const expressions::DateConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::FloatConstant *e1,
                             const expressions::FloatConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::BoolConstant *e1,
                             const expressions::BoolConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::StringConstant *e1,
                             const expressions::StringConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::DStringConstant *e1,
                             const expressions::DStringConstant *e2) = 0;
  virtual ProteusValue visit(const expressions::InputArgument *e1,
                             const expressions::InputArgument *e2) = 0;
  virtual ProteusValue visit(const expressions::RecordProjection *e1,
                             const expressions::RecordProjection *e2) = 0;
  virtual ProteusValue visit(const expressions::RecordConstruction *e1,
                             const expressions::RecordConstruction *e2) = 0;
  virtual ProteusValue visit(const expressions::IfThenElse *e1,
                             const expressions::IfThenElse *e2) = 0;
  virtual ProteusValue visit(const expressions::EqExpression *e1,
                             const expressions::EqExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::NeExpression *e1,
                             const expressions::NeExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::GeExpression *e1,
                             const expressions::GeExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::GtExpression *e1,
                             const expressions::GtExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::LeExpression *e1,
                             const expressions::LeExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::LtExpression *e1,
                             const expressions::LtExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::AddExpression *e1,
                             const expressions::AddExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::SubExpression *e1,
                             const expressions::SubExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::MultExpression *e1,
                             const expressions::MultExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::DivExpression *e1,
                             const expressions::DivExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::ModExpression *e1,
                             const expressions::ModExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::AndExpression *e1,
                             const expressions::AndExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::OrExpression *e1,
                             const expressions::OrExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::ProteusValueExpression *e1,
                             const expressions::ProteusValueExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::MinExpression *e1,
                             const expressions::MinExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::MaxExpression *e1,
                             const expressions::MaxExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::HashExpression *e1,
                             const expressions::HashExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::NegExpression *e1,
                             const expressions::NegExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::ExtractExpression *e1,
                             const expressions::ExtractExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::TestNullExpression *e1,
                             const expressions::TestNullExpression *e2) = 0;
  virtual ProteusValue visit(const expressions::CastExpression *e1,
                             const expressions::CastExpression *e2) = 0;
  virtual ~ExprTandemVisitor() {}
};

expression_t toExpression(Monoid m, expression_t lhs, expression_t rhs);

llvm::Constant *getIdentityElementIfSimple(Monoid m, const ExpressionType *type,
                                           Context *context);

inline expressions::EqExpression eq(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::NeExpression ne(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::LeExpression le(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::LtExpression lt(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::GeExpression ge(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::GtExpression gt(const expression_t &lhs,
                                    const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::OrExpression operator|(const expression_t &lhs,
                                           const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::AndExpression operator&(const expression_t &lhs,
                                            const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::AddExpression operator+(const expression_t &lhs,
                                            const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::SubExpression operator-(const expression_t &lhs,
                                            const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::NegExpression operator-(const expression_t &v) { return v; }

inline expressions::MultExpression operator*(const expression_t &lhs,
                                             const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::DivExpression operator/(const expression_t &lhs,
                                            const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::ModExpression operator%(const expression_t &lhs,
                                            const expression_t &rhs) {
  return {lhs, rhs};
}

inline expressions::RecordProjection expression_t::operator[](
    RecordAttribute proj) const {
  return {*this, proj};
}

inline expressions::RecordProjection expressions::InputArgument::operator[](
    RecordAttribute proj) const {
  // for (const auto &attr : projections) {
  //   if (attr.getAttrName() == proj.getAttrName()) {
  //     return {expression_t{*this}, attr};
  //   }
  // }
  // string error_msg = "[InputArgument: ] unknown projection";
  // LOG(ERROR) << error_msg;
  // throw runtime_error(error_msg);
  return expression_t{*this}[proj];
}

inline expression_t::expression_t(bool v)
    : expression_t(expressions::BoolConstant{v}) {}

inline expression_t::expression_t(int32_t v)
    : expression_t(expressions::IntConstant{v}) {}

inline expression_t::expression_t(int64_t v)
    : expression_t(expressions::Int64Constant{v}) {}

inline expression_t::expression_t(double v)
    : expression_t(expressions::FloatConstant{v}) {}

inline expression_t::expression_t(const char *v)
    : expression_t(std::string{v}) {}

inline expression_t::expression_t(std::string v)
    : expression_t(expressions::StringConstant{v}) {}

template <typename T>
constexpr bool compatibleExpr(const T *o1, const T *o2) {
  return true;
}

template <>
constexpr bool compatibleExpr<expressions::Constant>(
    const expressions::Constant *o1, const expressions::Constant *o2) {
  return o1->getConstantType() == o2->getConstantType();
}

template <typename T, typename Interface>
ProteusValue expressions::ExprVisitorVisitable<T, Interface>::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  auto r = dynamic_cast<const T *>(expr);
  if (r && compatibleExpr<Interface>(this, r)) {
    return v.visit(static_cast<const T *>(this), r);
  }
  string error_msg{"[Tandem Visitor: ] Incompatible Pair"};
  LOG(ERROR) << error_msg;
  throw runtime_error(error_msg);
}

template <>
ProteusValue expressions::
    ExprVisitorVisitable<expression_t, expressions::Expression>::acceptTandem(
        ExprTandemVisitor &v, const expressions::Expression *expr) const;

template <typename T, typename Interface>
ProteusValue expressions::ExprVisitorVisitable<T, Interface>::accept(
    ExprVisitor &v) const {
  return v.visit(static_cast<const T *>(this));
}

template <>
ProteusValue expressions::ExprVisitorVisitable<
    expression_t, expressions::Expression>::accept(ExprVisitor &v) const;

#endif /* EXPRESSIONS_HPP_ */
