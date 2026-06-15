/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_EXPRESSIONS_TEST_UTILS_HPP
#define PROTEUS_EXPRESSIONS_TEST_UTILS_HPP

#include <codegen/context/context.hpp>
#include <codegen/expressions/expressions.hpp>
#include <random>

namespace codegen {

/**
 * Shortcut to print the int32_t value using printi function
 * @param context Context with registered printi function
 * @param value int32_t value to print
 */
void printInt(Context *context, llvm::Value *value);

/**
 * Shortcut to print the int8_t value using printc function
 * @param context Context with registered printc function
 * @param value int8_t value to print
 */
void printChar(Context *context, llvm::Value *value);

/**
 * @class TestExpressionVisitor
 * is used ONLY in expressions tests to generate basic arithmetic expressions on
 * integers
 *
 * It contains functionality to generate +, -, *, / operations on integers.
 * Refer to the olap library to see the olap-specific visitor implementations.
 * @see DefaultExprVisitor, ExpressionGeneratorVisitor, ExpressionHasherVisitor,
 * ExpressionFlusherVisitor
 */
class TestExpressionVisitor : public ExprVisitor {
 public:
  /**
   * Construct the TestExpressionVisitor
   *
   * @param context Context with registered printi, printc functions
   * @param isPrintEnabled flag to turn on printing full arithmetic expression
   */
  explicit TestExpressionVisitor(Context *const context,
                                 bool isPrintEnabled = true)
      : context(context), isPrintEnabled(isPrintEnabled) {}

  ProteusValue visit(const expressions::IntConstant *e) override;

  ProteusValue visit(const expressions::Int64Constant *e) override {
    return {};
  }

  ProteusValue visit(const expressions::DateConstant *e) override { return {}; }

  ProteusValue visit(const expressions::FloatConstant *e) override {
    return {};
  }

  ProteusValue visit(const expressions::BoolConstant *e) override { return {}; }

  ProteusValue visit(const expressions::StringConstant *e) override {
    return {};
  }

  ProteusValue visit(const expressions::DStringConstant *e) override {
    return {};
  }

  ProteusValue visit(const expressions::InputArgument *e) override {
    return {};
  }

  ProteusValue visit(const expressions::RecordProjection *e) override {
    return {};
  }

  ProteusValue visit(const expressions::RecordConstruction *e) override {
    return {};
  }

  ProteusValue visit(const expressions::IfThenElse *e) override { return {}; }

  ProteusValue visit(const expressions::EqExpression *e) override { return {}; }

  ProteusValue visit(const expressions::NeExpression *e) override { return {}; }

  ProteusValue visit(const expressions::GeExpression *e) override { return {}; }

  ProteusValue visit(const expressions::GtExpression *e) override { return {}; }

  ProteusValue visit(const expressions::LeExpression *e) override { return {}; }

  ProteusValue visit(const expressions::LtExpression *e) override { return {}; }

  ProteusValue visit(const expressions::AddExpression *e) override;

  ProteusValue visit(const expressions::SubExpression *e) override;

  ProteusValue visit(const expressions::MultExpression *e) override;

  ProteusValue visit(const expressions::DivExpression *e) override;

  ProteusValue visit(const expressions::ModExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::AndExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::OrExpression *e) override { return {}; }

  ProteusValue visit(const expressions::PlaceholderExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::ProteusValueExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::MinExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::MaxExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::HashExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::RandExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::ExternExpression *e) override;

  ProteusValue visit(const expressions::HintExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::RefExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::AssignExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::NegExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::ExtractExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::TestNullExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::CastExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::ShiftLeftExpression *e) override {
    return {};
  }

  ProteusValue visit(
      const expressions::LogicalShiftRightExpression *e) override {
    return {};
  }

  ProteusValue visit(
      const expressions::ArithmeticShiftRightExpression *e) override {
    return {};
  }

  ProteusValue visit(const expressions::XORExpression *e) override {
    return {};
  }

 private:
  using ArithmeticOperation =
      std::function<llvm::Value *(llvm::Value *, llvm::Value *)>;

  Context *const context;
  const bool isPrintEnabled;

  void printInt(llvm::Value *value);

  void printChar(llvm::Value *value);

  /**
   * Generic function to implement simple binary operations
   *
   * @param e BinaryExpression to generate
   * @param operationChar operation's character to print between operands if
   * isPrintEnabled is true
   * @param operation the function that will be applied to the both operands of
   * e
   * @return result of the operation
   */
  ProteusValue generateArithmeticVisit(const expressions::BinaryExpression *e,
                                       char operationChar,
                                       const ArithmeticOperation &operation);
};

/**
 * Generate quadratic matrix filled with random elements of type T within the
 * range [from, to]
 *
 * Util function for the matrix multiplication test
 * @tparam T IntType that can be passed to the std::uniform_int_distribution
 * @tparam RowT type of the container that will be used as a storage of elements
 * with type T
 * @tparam MatrixT type of the container that will be used as a storage of RowT
 * rows
 * @param dimension size of the generated quadratic matrix
 * @param from lower bound of the generated values
 * @param to upper bound of the generated values
 * @return quadratic matrix of type MatrixT with random values of type T within
 * the range [from, to]
 */
template <typename T, typename RowT = std::vector<T>,
          typename MatrixT = std::vector<RowT>>
MatrixT GenerateRandomQuadraticMatrix(size_t dimension, T from, T to) {
  static std::random_device random_device;
  static std::mt19937 mt(random_device());
  std::uniform_int_distribution<T> distribution(from, to);

  MatrixT matrix;
  matrix.reserve(dimension);
  for (size_t rowIdx = 0; rowIdx < dimension; ++rowIdx) {
    RowT row;
    row.reserve(dimension);
    for (size_t columnIdx = 0; columnIdx < dimension; ++columnIdx) {
      row.push_back(distribution(mt));
    }
    matrix.push_back(std::move(row));
  }

  return matrix;
}

/**
 * Multiply matrixLhs and matrixRhs using naive matrix multiplication algorithm
 * and return the result of multiplication as a new matrix
 *
 * @tparam T type with a constructor from the integer type and implemented +, *
 * operators
 * @tparam RowT type of the container that is used as a storage of elements
 * with type T
 * @tparam MatrixT type of the container that is used as a storage of RowT
 * rows
 * @param matrixLhs left matrix in the multiplication
 * @param matrixRhs right matrix in the multiplication
 * @return result of the matrixLhs and matrixRhs multiplication
 */
template <typename T, typename RowT = std::vector<T>,
          typename MatrixT = std::vector<RowT>>
MatrixT MultiplyMatrices(const MatrixT &matrixLhs, const MatrixT &matrixRhs) {
  // Naive matrix multiplication
  const size_t matrixLhsHeight = matrixLhs.size();
  const size_t matrixLhsWidth =
      matrixLhsHeight > 0 ? matrixLhs.front().size() : 0;
  const size_t matrixRhsHeight = matrixRhs.size();
  const size_t matrixRhsWidth =
      matrixRhsHeight > 0 ? matrixRhs.front().size() : 0;

  assert(matrixLhsWidth == matrixRhsHeight &&
         "Matrices must have compatible sizes to be multiplied");
  assert(matrixLhsHeight > 0 && matrixRhsHeight > 0 &&
         "Matrices must have non-zero sizes");

  const size_t resultMatrixHeight = matrixLhsHeight;
  const size_t resultMatrixWidth = matrixRhsWidth;
  MatrixT resultMatrix;
  resultMatrix.reserve(resultMatrixHeight);

  for (size_t lhsRow = 0; lhsRow < matrixLhsHeight; ++lhsRow) {
    RowT resultRow;
    resultRow.reserve(resultMatrixWidth);

    for (size_t rhsColumn = 0; rhsColumn < matrixRhsWidth; ++rhsColumn) {
      T element = {0};
      for (size_t lhsColumn = 0; lhsColumn < matrixLhsWidth; ++lhsColumn) {
        element = element + matrixLhs[lhsRow][lhsColumn] *
                                matrixRhs[lhsColumn][rhsColumn];
      }
      resultRow.push_back(std::move(element));
    }

    resultMatrix.push_back(std::move(resultRow));
  }

  return resultMatrix;
}

using ExpressionsRow = std::vector<expression_t>;
using ExpressionsMatrix = std::vector<ExpressionsRow>;

/**
 * Convert matrix with type MatrixT which stores elements of type T into
 * ExpressionMatrix
 *
 * @tparam T type from which expression_t must have a constructor
 * @tparam RowT type of the container that is used as a storage of elements
 * with type T
 * @tparam MatrixT type of the container that is used as a storage of RowT
 * rows
 * @param matrix matrix with elements with type T that will be converted into
 * ExpressionMatrix
 * @return ExpressionMatrix with expression_t constructed with the corresponding
 * elements from the input matrix
 */
template <typename T, typename RowT = std::vector<T>,
          typename MatrixT = std::vector<RowT>>
ExpressionsMatrix GetExpressionMatrixFromMatrix(const MatrixT &matrix) {
  ExpressionsMatrix expressionsMatrix;
  expressionsMatrix.reserve(matrix.size());

  for (const auto &row : matrix) {
    ExpressionsRow expressionsRow;
    expressionsRow.reserve(row.size());

    for (const auto &item : row) {
      expressionsRow.emplace_back(item);
    }

    expressionsMatrix.push_back(std::move(expressionsRow));
  }

  return expressionsMatrix;
}

}  // namespace codegen

#endif  // PROTEUS_EXPRESSIONS_TEST_UTILS_HPP
