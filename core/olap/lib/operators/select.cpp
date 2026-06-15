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

#include "select.hpp"

#include <olap/util/jit/control-flow/if-statement.hpp>

void Select::produce_(OlapParallelContext *context) {
  getChild()->produce(context);
}

void Select::consume(OlapParallelContext *context,
                     const OperatorState &childState) {
  gen_if(expr, childState, context)([&] {
    // Triggering parent
    OperatorState newState{*this, childState};
    getParent()->consume(context, newState);
  });
}
