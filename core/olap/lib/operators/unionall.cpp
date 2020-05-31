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

#include "unionall.hpp"

void UnionAll::produce_(ParallelContext *context) {
  generate_catch(context);

  catch_pip = context->operator->();

  // push new pipeline for the throw part
  for (const auto &child : children) {
    context->popPipeline();
    context->pushPipeline();

    context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
    context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

    child->produce(context);
  }
}
