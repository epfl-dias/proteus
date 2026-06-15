/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2025
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

#ifndef PROTEUS_SSB_SYNTHETIC_HPP
#define PROTEUS_SSB_SYNTHETIC_HPP

#include <olap/plan/prepared-statement.hpp>
#include <olap/routing/routing-policy-types.hpp>
#include <query-shaping/nvme-shapers.hpp>
#include <query-shaping/query-shaper.hpp>

struct SynthJoinQueryArgs {
  std::shared_ptr<proteus::GPUOnlyNVMeMorsel> morph = nullptr;
  int scan_memmove_dop_per_gpu = 12;
  int scan_memmove_slack = 4;
  int scan_router_slack = 4;
  int lo_quantity_le_predicate;
};

PreparedStatement synthetic_join_zero(SynthJoinQueryArgs);
PreparedStatement synthetic_join_one(SynthJoinQueryArgs);
PreparedStatement synthetic_join_two(SynthJoinQueryArgs);
PreparedStatement synthetic_join_three(SynthJoinQueryArgs);
PreparedStatement synthetic_join_four(SynthJoinQueryArgs);

#endif  // PROTEUS_SSB_SYNTHETIC_HPP
