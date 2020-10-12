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

#include "glo.hpp"

#include <iostream>

#include "utils/utils.hpp"

uint g_num_partitions = 1;
uint g_delta_size = 4;

bool timed_func::terminate = false;
int timed_func::num_active_runners = 0;

namespace global_conf {
proteus::utils::Percentile read_cdf("read_cdf");
proteus::utils::Percentile read_mv_cdf("read_mv_cdf");
proteus::utils::Percentile update_cdf("update_cdf");
proteus::utils::Percentile insert_cdf("insert_cdf");
}  // namespace global_conf
