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

#include "operators/split.hpp"

#include <cstring>

#include "expressions/expressions-generator.hpp"

void Split::produce_(ParallelContext *context) {
  UnaryOperator::setParent(parent[produce_calls]);
  generate_catch(context);

  catch_pip.push_back(context->operator->());

  if (++produce_calls != fanout) return;

  context->popPipeline();

  // push new pipeline for the throw part
  context->pushPipeline();

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

void Split::open(Pipeline *pip) {
  // time_block t("Tinit_exchange: ");

  std::lock_guard<std::mutex> guard(init_mutex);

  if (firers.empty()) {
    remaining_producers = producers;
    for (int i = 0; i < fanout; ++i) {
      firers.emplace_back(&Split::fire, this, i, catch_pip[i]);
    }
  }
}
