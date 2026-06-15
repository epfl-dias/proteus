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

#ifndef IF_STATEMENT_HPP_
#define IF_STATEMENT_HPP_

#include <codegen/context/context.hpp>
#include <codegen/expressions/expressions.hpp>
#include <codegen/jit/control-flow/if-statement.hpp>
#include <olap/operators/operator-state.hpp>

if_branch gen_if(const expression_t &expr, const OperatorState &state,
                 Context *context);

#endif /* IF_STATEMENT_HPP_ */
