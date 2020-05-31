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

#ifndef GPU_MATERIALIZER_EXPR_HPP_
#define GPU_MATERIALIZER_EXPR_HPP_

#include "olap/expressions/expressions.hpp"

struct GpuMatExpr {
 public:
  expression_t expr;
  size_t packet;
  size_t bitoffset;
  size_t packind;

  GpuMatExpr(expression_t expr, size_t packet, size_t bitoffset)
      : expr(expr), packet(packet), bitoffset(bitoffset), packind(-1) {}
};

#endif /* GPU_MATERIALIZER_EXPR_HPP_ */
