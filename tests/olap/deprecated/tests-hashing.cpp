/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include <platform/common/common.hpp>

#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "gtest/gtest.h"
#include "operators/join.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/reduce.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>

template <class T>
inline void my_hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v);
}

TEST(Hashing, IntegerNotGenerated) {
  Context &ctx = *prepareContext("HashInt");
  Catalog &catalog = Catalog::getInstance();

  std::hash<int> hasher;
  size_t seed = 0;

  hash_combine(seed, 15);
  hash_combine(seed, 20);
  hash_combine(seed, 29);
  cout << "Seed 1: " << seed << endl;

  seed = 0;
  size_t seedPartial = 0;
  hash_combine(seed, 15);

  hash_combine(seedPartial, 20);
  hash_combine(seedPartial, 29);
  hash_combine(seed, seedPartial);
  cout << "Seed 2: " << seed << endl;

  seed = 0;
  my_hash_combine(seed, 20);
  my_hash_combine(seed, 25);
  cout << "Seed A: " << seed << endl;

  seed = 0;
  my_hash_combine(seed, 25);
  my_hash_combine(seed, 20);
  cout << "Seed B: " << seed << endl;

  EXPECT_TRUE(true);
}

TEST(Hashing, Constants) {
  Context &ctx = *prepareContext("HashConstants");
  Catalog &catalog = Catalog::getInstance();

  Root rootOp = Root(nullptr);
  map<RecordAttribute, ProteusValueMemory> varPlaceholder;
  OperatorState statePlaceholder = OperatorState(rootOp, varPlaceholder);
  ExpressionHasherVisitor hasher =
      ExpressionHasherVisitor(&ctx, statePlaceholder);

  int inputInt = 1400;
  double inputFloat = 1300.5;
  bool inputBool = true;
  string inputString = string("1400");

  expressions::IntConstant *val_int = new expressions::IntConstant(inputInt);
  hasher.visit(val_int);
  expressions::FloatConstant *val_float =
      new expressions::FloatConstant(inputFloat);
  hasher.visit(val_float);
  expressions::BoolConstant *val_bool =
      new expressions::BoolConstant(inputBool);
  hasher.visit(val_bool);
  expressions::StringConstant *val_string =
      new expressions::StringConstant(inputString);
  hasher.visit(val_string);
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Non-generated counterparts
  std::hash<int> hasherInt;
  std::hash<double> hasherFloat;
  std::hash<bool> hasherBool;
  std::hash<string> hasherString;

  cout << "[Int - not generated:] " << hasherInt(inputInt) << endl;
  cout << "[Float - not generated:] " << hasherFloat(inputFloat) << endl;
  cout << "[Float - not generated:] " << hasherFloat(1.300500e+03) << endl;

  cout << "[Bool - not generated:] " << hasherBool(inputBool) << endl;
  cout << "[String - not generated:] " << hasherString(inputString) << endl;

  EXPECT_TRUE(true);
}

TEST(Hashing, BinaryExpressions) {
  Context &ctx = *prepareContext("HashBinaryExpressions");
  Catalog &catalog = Catalog::getInstance();

  Root rootOp = Root(nullptr);
  map<RecordAttribute, ProteusValueMemory> varPlaceholder;
  OperatorState statePlaceholder = OperatorState(rootOp, varPlaceholder);
  ExpressionHasherVisitor hasher =
      ExpressionHasherVisitor(&ctx, statePlaceholder);

  int inputInt = 1400;
  double inputFloat = 1300.5;
  bool inputBool = true;
  string inputString = string("1400");

  expressions::IntConstant *val_int = new expressions::IntConstant(inputInt);
  expressions::FloatConstant *val_float =
      new expressions::FloatConstant(inputFloat);
  expressions::BoolConstant *val_bool =
      new expressions::BoolConstant(inputBool);
  expressions::StringConstant *val_string =
      new expressions::StringConstant(inputString);

  expressions::NeExpression *bool_ne =
      new expressions::NeExpression(val_int, val_int);
  expressions::EqExpression *bool_eq =
      new expressions::EqExpression(val_float, val_float);
  expressions::AddExpression *int_add =
      new expressions::AddExpression(val_int, val_int);
  expressions::MultExpression *float_mult =
      new expressions::MultExpression(val_float, val_float);

  hasher.visit(bool_ne);
  hasher.visit(bool_eq);
  hasher.visit(int_add);
  hasher.visit(float_mult);

  ctx.prepareFunction(ctx.getGlobalFunction());

  // Non-generated counterparts
  std::hash<int> hasherInt;
  std::hash<double> hasherFloat;
  std::hash<bool> hasherBool;

  cout << "[Bool - not generated:] " << hasherBool(inputInt != inputInt)
       << endl;
  cout << "[Bool - not generated:] " << hasherBool(inputFloat == inputFloat)
       << endl;
  cout << "[Int - not generated:] " << hasherInt(inputInt + inputInt) << endl;
  cout << "[Float - not generated:] " << hasherFloat(inputFloat * inputFloat);
  EXPECT_TRUE(true);
}

TEST(Hashing, IfThenElse) {
  Context &ctx = *prepareContext("HashIfThenElse");
  Catalog &catalog = Catalog::getInstance();

  Root rootOp = Root(nullptr);
  map<RecordAttribute, ProteusValueMemory> varPlaceholder;
  OperatorState statePlaceholder = OperatorState(rootOp, varPlaceholder);
  ExpressionHasherVisitor hasher =
      ExpressionHasherVisitor(&ctx, statePlaceholder);

  int inputInt = 1400;
  double inputFloat = 1300.5;
  bool inputBool = true;
  string inputString = string("1400");

  expressions::IntConstant *val_int = new expressions::IntConstant(inputInt);
  expressions::FloatConstant *val_float =
      new expressions::FloatConstant(inputFloat);
  expressions::BoolConstant *val_bool =
      new expressions::BoolConstant(inputBool);
  expressions::StringConstant *val_string =
      new expressions::StringConstant(inputString);

  expressions::EqExpression *bool_eq =
      new expressions::EqExpression(val_float, val_float);
  expressions::AddExpression *int_add =
      new expressions::AddExpression(val_int, val_int);
  expressions::SubExpression *int_sub =
      new expressions::SubExpression(val_int, val_int);

  expressions::Expression *ifElse =
      new expressions::IfThenElse(bool_eq, int_add, int_sub);
  ifElse->accept(hasher);

  ctx.prepareFunction(ctx.getGlobalFunction());

  // Non-generated counterparts
  std::hash<int> hasherInt;

  int toHash =
      inputFloat == inputFloat ? inputInt + inputInt : inputInt - inputInt;
  cout << "[Int - not generated:] " << hasherInt(toHash) << endl;
  EXPECT_TRUE(true);
}

// Step 3. Call RUN_ALL_TESTS() in main().
//
// We do this by linking in src/gtest_main.cc file, which consists of
// a main() function which calls RUN_ALL_TESTS() for us.
//
// This runs all the tests you've defined, prints the result, and
// returns 0 if successful, or 1 otherwise.
//
// Did you notice that we didn't register the tests?  The
// RUN_ALL_TESTS() macro magically knows about all the tests we
// defined.  Isn't this convenient?
