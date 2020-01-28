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

#ifndef EXPRESSIONTYPES_HPP_
#define EXPRESSIONTYPES_HPP_

#include <cassert>
#include <memory>
#include <ostream>

#include "common/common.hpp"

/* Original.*/
// enum typeID    { BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET, BLOCK };
/* Extended due to caching (for now)*/
enum typeID {
  BOOL,
  DSTRING,
  STRING,
  FLOAT,
  INT,
  DATE,
  RECORD,
  LIST,
  BAG,
  SET,
  INT64,
  COMPOSITE,
  BLOCK
};

class ExprTypeVisitor;

namespace llvm {
class Type;
class LLVMContext;
}  // namespace llvm

class ExpressionType {
 public:
  virtual string getType() const = 0;
  virtual typeID getTypeID() const = 0;
  virtual ~ExpressionType() {}
  virtual bool isPrimitive() const = 0;
  virtual llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const {
    string error_msg =
        string("Type " + getType() + " is not mapped into an LLVM-type.");
    LOG(INFO) << error_msg;
    // throw runtime_error(error_msg);
    return nullptr;
  }

  virtual void accept(ExprTypeVisitor &v) const = 0;
};

template <typename T, typename Interface = ExpressionType>
class ExpressionTypeVisitable : public Interface {
  using Interface::Interface;
  virtual void accept(ExprTypeVisitor &v) const;
};

class PrimitiveType : public ExpressionType {
 public:
  virtual llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const = 0;
};

template <typename T, typeID id>
class PrimitiveTypeCRTP : public ExpressionTypeVisitable<T, PrimitiveType> {
 public:
  std::string getType() const { return T::name; }
  typeID getTypeID() const { return id; }
  bool isPrimitive() const { return true; }
};

class BoolType : public PrimitiveTypeCRTP<BoolType, BOOL> {
 public:
  static constexpr auto name = "Bool";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class StringType : public PrimitiveTypeCRTP<StringType, STRING> {
 public:
  static constexpr auto name = "String";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class DStringType : public PrimitiveTypeCRTP<DStringType, DSTRING> {
 public:
  static constexpr auto name = "DString";

  DStringType(void *dictionary = nullptr) : dictionary(dictionary) {}

  void *getDictionary() const {
    assert(dictionary);
    return dictionary;
  }

  void setDictionary(void *dict) {
    assert(dict);
    dictionary = dict;
  }

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;

 private:
  void *dictionary;
};

class FloatType : public PrimitiveTypeCRTP<FloatType, FLOAT> {
 public:
  static constexpr auto name = "Float";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class IntType : public PrimitiveTypeCRTP<IntType, INT> {
 public:
  static constexpr auto name = "Int";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class Int64Type : public PrimitiveTypeCRTP<Int64Type, INT64> {
 public:
  static constexpr auto name = "Int64";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

/**
 * Represented as 64bit timestamps, in __msec__ (UTC) since (UTC) epoch time
 *
 * Conversion in locale should be handled by the plugin during reading and by
 * the query parser!
 */
class DateType : public PrimitiveTypeCRTP<DateType, DATE> {
 public:
  static constexpr auto name = "Date";

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class CollectionType : public ExpressionType {
 public:
  CollectionType(const ExpressionType &type) : type(type) {}

  virtual bool isPrimitive() const final { return false; }
  const ExpressionType &getNestedType() const { return type; }
  virtual ~CollectionType() = default;

 private:
  const ExpressionType &type;
};

template <typename T, typeID id>
class CollectionTypeCRTP : public ExpressionTypeVisitable<T, CollectionType> {
 protected:
  using ExpressionTypeVisitable<T, CollectionType>::ExpressionTypeVisitable;

 public:
  virtual typeID getTypeID() const { return id; }

  virtual string getType() const {
    return T::name + std::string("(") + this->getNestedType().getType() +
           string(")");
  }
};

class BlockType : public CollectionTypeCRTP<BlockType, BLOCK> {
 public:
  static constexpr auto name = "Block";
  using CollectionTypeCRTP::CollectionTypeCRTP;

  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;
};

class ListType : public CollectionTypeCRTP<ListType, LIST> {
 public:
  static constexpr auto name = "List";
  using CollectionTypeCRTP::CollectionTypeCRTP;
};

class BagType : public CollectionTypeCRTP<BagType, BAG> {
 public:
  static constexpr auto name = "Bag";
  using CollectionTypeCRTP::CollectionTypeCRTP;
};

class SetType : public CollectionTypeCRTP<SetType, SET> {
 public:
  static constexpr auto name = "Set";
  using CollectionTypeCRTP::CollectionTypeCRTP;
};

class RecordAttribute {
 public:
  RecordAttribute()
      : relName(""), attrName(""), type(nullptr), attrNo(-1), projected(false) {
    cout << "ANONYMOUS CONSTRUCTOR!!" << endl;
  }
  RecordAttribute(int no, string relName, string attrName,
                  const ExpressionType *type, bool make_block = false)
      : relName(relName),
        attrName(attrName),
        originalRelName(relName),
        type(make_block ? new BlockType(*type) : type),
        attrNo(no),
        projected(false) {
    if (relName == "") {
      string error_msg =
          string("Unexpected, no-relname attribute: ") + attrName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  RecordAttribute(int no, string relName, const char *attrName,
                  const ExpressionType *type)
      : relName(relName),
        originalRelName(relName),
        type(type),
        attrNo(no),
        projected(false) {
    this->attrName = string(attrName);
    if (relName == "") {
      string error_msg =
          string("Unexpected, no-relname attribute: ") + attrName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  RecordAttribute(int no, string originalRelName, string relName,
                  string attrName, const ExpressionType *type)
      : relName(relName),
        attrName(attrName),
        originalRelName(originalRelName),
        type(type),
        attrNo(no),
        projected(false) {
    if (relName == "") {
      string error_msg =
          string("Unexpected, no-relname attribute: ") + attrName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    if (originalRelName == "") {
      string error_msg =
          string("Unexpected, no-origrelname attribute: ") + attrName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  // Constructor used STRICTLY for comparisons in maps
  //    RecordAttribute(string relName, string attrName)
  //            : attrNo(-1), relName(relName), attrName(attrName),
  //            type(nullptr), projected(false)     {}

  /* OID Type needed so that we know what we materialize
   * => Subsequent functions / programs use info to parse caches */
  RecordAttribute(string relName, string attrName, const ExpressionType *type)
      : relName(relName),
        attrName(attrName),
        originalRelName(relName),
        type(type),
        attrNo(-1),
        projected(false) {
    // cout << "RELNAME:[" << relName << "]" << endl;
    if (relName == "") {
      string error_msg =
          string("Unexpected, no-relname attribute: ") + attrName;
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  RecordAttribute(const RecordAttribute &obj, bool make_block)
      : type(make_block ? new BlockType(*(obj.getOriginalType()))
                        : obj.getOriginalType()) {
    this->attrNo = obj.attrNo;
    this->attrName = obj.attrName;
    this->relName = obj.relName;
    this->originalRelName = obj.originalRelName;
    this->projected = obj.projected;
  }

  //    RecordAttribute(const RecordAttribute& obj) :
  //    type(obj.getOriginalType()) {
  //        this->attrNo = obj.attrNo;
  //        this->relName = obj.attrName;
  //        this->projected = obj.projected;
  //    }

  string getType() const { return attrName + " " + type->getType(); }
  const ExpressionType *getOriginalType() const { return type; }
  llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const {
    return getOriginalType()->getLLVMType(ctx);
  }
  string getName() const { return attrName; }
  string getRelationName() const { return relName; }
  string getOriginalRelationName() const { return originalRelName; }
  string getAttrName() const { return attrName; }
  // CONVENTION: Fields requested can be 1-2-3-etc.
  int getAttrNo() const { return attrNo; }
  void setProjected() { projected = true; }
  bool isProjected() { return projected; }

 private:
  string relName;
  string attrName;
  string originalRelName;
  const ExpressionType *type;
  // Atm, used by eager CSV plugin (for the native relations)
  int attrNo;
  bool projected;
};

std::ostream &operator<<(std::ostream &o, const RecordAttribute &rec);

std::ostream &operator<<(std::ostream &o, const ExpressionType &rec);

class RecordType : public ExpressionTypeVisitable<RecordType, ExpressionType> {
 public:
  RecordType() {}
  RecordType(list<RecordAttribute *> args) : args(args) {
    for (const auto &arg : this->args) {
      argsMap[arg->getAttrName()] = arg;
    }
  }

  RecordType(vector<RecordAttribute *> args) {
    for (const auto &arg : args) {
      auto new_arg = new RecordAttribute{*arg};
      this->args.push_back(new_arg);
      argsMap[arg->getAttrName()] = new_arg;
    }
  }

  RecordType(const RecordType &rec) {
    argsMap.insert(rec.argsMap.begin(), rec.argsMap.end());
    args.insert(args.begin(), rec.args.begin(), rec.args.end());
  }

  string getType() const {
    stringstream ss;
    ss << "Record(";
    int count = 0;
    int size = args.size();
    for (list<RecordAttribute *>::const_iterator it = args.begin();
         it != args.end(); it++) {
      ss << (*it)->getType();
      count++;
      if (count != size) {
        ss << ", ";
      }
    }
    ss << ")";
    return ss.str();
  }

  virtual llvm::Type *getLLVMType(llvm::LLVMContext &ctx) const;

  typeID getTypeID() const { return RECORD; }
  list<RecordAttribute *> getArgs() const { return args; }
  map<string, RecordAttribute *> &getArgsMap() { return argsMap; }
  int getArgsNo() const { return args.size(); }
  bool isPrimitive() const { return false; }
  ~RecordType() {}

  void appendAttribute(RecordAttribute *attr) {
#ifndef NDEBUG
    auto inserted =
#endif
        argsMap.emplace(attr->getAttrName(), attr);
    if (!inserted.second) {
      // assert(*(attr->getType()) == inserted.first->getTypeID());
      return;
    }
    // assert(inserted.second && "Attribute already exists!");
    args.push_back(attr);
  }

  const RecordAttribute *getArg(string name) const {
    auto r = argsMap.find(name);
    if (r == argsMap.end()) return nullptr;
    return r->second;
  }

  int getIndex(RecordAttribute *x) const;

 private:
  list<RecordAttribute *> args;
  map<string, RecordAttribute *> argsMap;
};

class ExprTypeVisitor {
 public:
  virtual void visit(const IntType &type) = 0;
  virtual void visit(const Int64Type &type) = 0;
  virtual void visit(const BoolType &type) = 0;
  virtual void visit(const FloatType &type) = 0;
  virtual void visit(const DateType &type) = 0;
  virtual void visit(const StringType &type) = 0;
  virtual void visit(const DStringType &type) = 0;
  virtual void visit(const RecordType &type) = 0;
  virtual void visit(const SetType &type) = 0;
  virtual void visit(const BlockType &type) = 0;
  virtual void visit(const BagType &type) = 0;
  virtual void visit(const ListType &type) = 0;
  virtual ~ExprTypeVisitor() = default;
};

template <typename T, typename Interface>
void ExpressionTypeVisitable<T, Interface>::accept(ExprTypeVisitor &v) const {
  v.visit(*static_cast<const T *>(this));
}

/* XXX Not too sure these comparators make sense.
 * If difference between hashed expressions boils down to this
 * point, I am doing sth wrong. */
inline bool operator<(const ExpressionType &l, const ExpressionType &r) {
  cout << "Comparing GENERIC EXPRESSION TYPE" << endl;
  return l.getType() < r.getType();
}

bool recordComparator(RecordAttribute *x, RecordAttribute *y);
inline bool operator<(const RecordAttribute &l, const RecordAttribute &r) {
  if (l.getRelationName() == r.getRelationName()) {
    return l.getAttrName() < r.getAttrName();
  } else {
    return l.getRelationName() < r.getRelationName();
  }
}
inline bool operator==(const RecordAttribute &l, const RecordAttribute &r) {
  return (l.getRelationName() == r.getRelationName()) &&
         (l.getAttrName() == r.getAttrName());
}
//
//
// inline bool operator<(const RecordType& l, const RecordType& r) {
//    list<RecordAttribute*>& leftArgs = l.getArgs();
//    list<RecordAttribute*>& rightArgs = r.getArgs();
//
//    if (leftArgs.size() != rightArgs.size()) {
//        return leftArgs.size() < rightArgs.size();
//    }
//
//    list<RecordAttribute*>::iterator itLeftArgs = leftArgs.begin();
//    list<RecordAttribute*>::iterator itRightArgs = rightArgs.begin();
//
//    while (itLeftArgs != leftArgs.end()) {
//        RecordAttribute attrLeft = *(*itLeftArgs);
//        RecordAttribute attrRight = *(*itRightArgs);
//
//        bool eqAttr = !(attrLeft < attrRight) && !(attrRight < attrLeft);
//        if (!eqAttr) {
//            return attrLeft < attrRight;
//        }
//        itLeftArgs++;
//        itRightArgs++;
//    }
//    return false;
//}
//;

#endif /* EXPRESSIONTYPES_HPP_ */
