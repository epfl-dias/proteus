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

#ifndef PROTEUS_GPU_AGGR_MAT_EXPR_HPP
#define PROTEUS_GPU_AGGR_MAT_EXPR_HPP

#include <olap/expressions/expressions.hpp>
#include <olap/operators/monoids.hpp>

class GpuAggrMatExpr {
 public:
  expression_t expr;
  size_t packet;
  size_t bitoffset;
  size_t packind;
  Monoid m;
  bool is_m;

  GpuAggrMatExpr(expression_t expr, size_t packet, size_t bitoffset, Monoid m)
      : expr(expr),
        packet(packet),
        bitoffset(bitoffset),
        packind(-1),
        m(m),
        is_m(true) {}

  GpuAggrMatExpr(expression_t expr, size_t packet, size_t bitoffset)
      : expr(expr),
        packet(packet),
        bitoffset(bitoffset),
        packind(-1),
        m(SUM),
        is_m(false) {}

  bool is_aggregation() { return is_m; }
};

#endif /* PROTEUS_GPU_AGGR_MAT_EXPR_HPP */
