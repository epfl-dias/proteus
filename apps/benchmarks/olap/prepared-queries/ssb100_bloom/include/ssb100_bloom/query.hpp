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

#ifndef PROTEUS_SSB100_BLOOM_QUERY_HPP
#define PROTEUS_SSB100_BLOOM_QUERY_HPP

#include <olap/plan/prepared-statement.hpp>

enum SLAZY {
  BLOOM_CPUFILTER_PROJECT,
  BLOOM_CPUFILTER_NOPROJECT,
  BLOOM_GPUFILTER_NOPROJECT,
};

namespace ssb100_bloom {
class Query {
 public:
  PreparedStatement prepare11(bool memmv, SLAZY conf, size_t bloomSize);
  PreparedStatement prepare11_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare12(bool memmv, SLAZY conf, size_t bloomSize);
  PreparedStatement prepare12_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare13(bool memmv, SLAZY conf, size_t bloomSize);
  PreparedStatement prepare13_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare21(bool memmv, size_t bloomSize);
  PreparedStatement prepare21_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare22(bool memmv, size_t bloomSize);
  PreparedStatement prepare22_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare23(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_b2(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_c(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_d(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_e(bool memmv, size_t bloomSize);
  PreparedStatement prepare23_mat(bool memmv, size_t bloomSize);
  PreparedStatement prepare31(bool memmv, size_t bloomSize);
  PreparedStatement prepare31_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare32(bool memmv, size_t bloomSize);
  PreparedStatement prepare32_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare33(bool memmv, size_t bloomSize);
  PreparedStatement prepare33_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare34(bool memmv, size_t bloomSize);
  PreparedStatement prepare34_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare41(bool memmv, size_t bloomSize);
  PreparedStatement prepare41_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare41_c(bool memmv, size_t bloomSize);
  PreparedStatement prepare42(bool memmv, size_t bloomSize);
  PreparedStatement prepare42_b(bool memmv, size_t bloomSize);
  PreparedStatement prepare43(bool memmv, size_t bloomSize);
  PreparedStatement prepare43_b(bool memmv, size_t bloomSize);
  //  PreparedStatement prepare43(bool memmv);

  std::vector<PreparedStatement> prepareAll(bool memmv, SLAZY conf,
                                            size_t bloomSize) {
    return {
        // Q1.*
        prepare11(memmv, conf, bloomSize), prepare12(memmv, conf, bloomSize),
        prepare13(memmv, conf, bloomSize),
        // Q2.*
        prepare21(memmv, bloomSize), prepare22(memmv, bloomSize),
        prepare42(memmv, bloomSize),
        // Q3.*
        prepare31(memmv, bloomSize), prepare32(memmv, bloomSize),
        prepare33(memmv, bloomSize), prepare34(memmv, bloomSize),
        // Q4.*
        prepare41(memmv, bloomSize), prepare42(memmv, bloomSize),
        prepare42(memmv, bloomSize), prepare43(memmv, bloomSize)
        // EOQ
    };
  }
};
};  // namespace ssb100_bloom

#endif /* PROTEUS_SSB100_BLOOM_QUERY_HPP */
