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

#ifndef HTAP_CH_QUERIES_HPP_
#define HTAP_CH_QUERIES_HPP_

#include <string>

#include "../queries.hpp"
#include "queries/query-interface.hpp"

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

using default_plugin_t = AeolusRemotePlugin;

#include "micro-qstock.hpp"
#include "micro-sum.hpp"
#include "q01.hpp"
#include "q04.hpp"
#include "q06.hpp"
#include "q08.hpp"
#include "q12.hpp"
#include "q18.hpp"
#include "q19.hpp"

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_sum_c1t();

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch_c1t();

// // template <typename Tplugin>
// // PreparedStatement q_ch2_c1t();

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch1_c1t();

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch6_c1t();

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch19_c1t();

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_sum_cpar(
//     DegreeOfParallelism dop,
//     std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//     std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch_cpar(DegreeOfParallelism dop,
//                             std::unique_ptr<Affinitizer> aff_parallel =
//                             nullptr, std::unique_ptr<Affinitizer> aff_reduce
//                             = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch1_cpar(
//     DegreeOfParallelism dop,
//     std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//     std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch6_cpar(
//     DegreeOfParallelism dop,
//     std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//     std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch19_cpar(DegreeOfParallelism dop,
//                               std::unique_ptr<Affinitizer> aff_parallel,
//                               std::unique_ptr<Affinitizer> aff_parallel2,
//                               std::unique_ptr<Affinitizer> aff_reduce);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_sum(DegreeOfParallelism dop,
//                         std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//                         std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch(DegreeOfParallelism dop,
//                        std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//                        std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <typename Tplugin = default_plugin_t>
// PreparedStatement q_ch4(DegreeOfParallelism dop,
//                         std::unique_ptr<Affinitizer> aff_parallel = nullptr,
//                         std::unique_ptr<Affinitizer> aff_reduce = nullptr);

// template <>
// template <typename Tplugin>
// PreparedStatement Q<18>::c1t() {
//   return q_ch18_c1t<Tplugin>();
// }

#endif /* HTAP_CH_QUERIES_HPP_ */
