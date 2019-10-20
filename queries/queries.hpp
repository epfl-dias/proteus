/*
    Harmonia -- High-performance elastic HTAP on heterogeneous hardware.

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

#include <string>

#include "codegen/plan/prepared-statement.hpp"
#include "routing/affinitizers.hpp"
#include "routing/degree-of-parallelism.hpp"

extern std::string tpcc_orderline;

extern std::string ol_o_id;
extern std::string ol_d_id;
extern std::string ol_w_id;
extern std::string ol_number;
extern std::string ol_i_id;
extern std::string ol_supply_w_id;
extern std::string ol_delivery_d;
extern std::string ol_quantity;
extern std::string ol_amount;
extern std::string ol_dist_info;

extern std::string tpcc_order;

extern std::string o_id;
extern std::string o_d_id;
extern std::string o_w_id;
extern std::string o_c_id;
extern std::string o_entry_d;
extern std::string o_carrier_id;
extern std::string o_ol_cnt;
extern std::string o_all_local;

PreparedStatement q_sum_c1t();
PreparedStatement q_ch_c1t();
PreparedStatement q_ch2_c1t();
PreparedStatement q_ch1_c1t();
PreparedStatement q_ch6_c1t();
PreparedStatement q_ch19_c1t();

PreparedStatement q_sum_cpar(
    DegreeOfParallelism dop,
    std::unique_ptr<Affinitizer> aff_parallel = nullptr,
    std::unique_ptr<Affinitizer> aff_reduce = nullptr);
PreparedStatement q_ch_cpar(DegreeOfParallelism dop,
                            std::unique_ptr<Affinitizer> aff_parallel = nullptr,
                            std::unique_ptr<Affinitizer> aff_reduce = nullptr);
PreparedStatement q_ch1_cpar(
    DegreeOfParallelism dop,
    std::unique_ptr<Affinitizer> aff_parallel = nullptr,
    std::unique_ptr<Affinitizer> aff_reduce = nullptr);
PreparedStatement q_ch6_cpar(
    DegreeOfParallelism dop,
    std::unique_ptr<Affinitizer> aff_parallel = nullptr,
    std::unique_ptr<Affinitizer> aff_reduce = nullptr);
PreparedStatement q_ch19_cpar(DegreeOfParallelism dop,
                              std::unique_ptr<Affinitizer> aff_parallel,
                              std::unique_ptr<Affinitizer> aff_parallel2,
                              std::unique_ptr<Affinitizer> aff_reduce);

PreparedStatement q_sum(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel = nullptr,
                        std::unique_ptr<Affinitizer> aff_reduce = nullptr);
PreparedStatement q_ch(DegreeOfParallelism dop,
                       std::unique_ptr<Affinitizer> aff_parallel = nullptr,
                       std::unique_ptr<Affinitizer> aff_reduce = nullptr);

template <typename Tp, typename Tr>
PreparedStatement q_ch1(DegreeOfParallelism dop, Tp aff_parallel,
                        Tr aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch1_c1t();
  return q_ch1_cpar(dop, aff_parallel(), aff_reduce());
}
PreparedStatement q_ch4(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel = nullptr,
                        std::unique_ptr<Affinitizer> aff_reduce = nullptr);
template <typename Tp, typename Tr>
PreparedStatement q_ch6(DegreeOfParallelism dop, Tp aff_parallel,
                        Tr aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch6_c1t();
  return q_ch6_cpar(dop, aff_parallel(), aff_reduce());
}
template <typename Tp, typename Tr>
PreparedStatement q_ch19(DegreeOfParallelism dop, Tp aff_parallel,
                         Tr aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch19_c1t();
  return q_ch19_cpar(dop, aff_parallel(), aff_parallel(), aff_reduce());
}
