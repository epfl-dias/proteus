/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <fstream>
#include <platform/util/time-registry.hpp>

const TimeRegistry::Key TimeRegistry::Ignore{"Ignore"};

TimeRegistry::~TimeRegistry() {
  std::ofstream timeout{"time-registry.txt"};
  for (const auto &rec : registry) {
    timeout << "Total time in " << rec.first.getKey() << ": "
            << std::chrono::duration_cast<std::chrono::milliseconds>(rec.second)
                   .count()
            << "ms\n";
  }
}
