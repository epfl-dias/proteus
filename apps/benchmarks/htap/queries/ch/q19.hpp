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

#ifndef HARMONIA_QUERIES_CH_Q19_HPP_
#define HARMONIA_QUERIES_CH_Q19_HPP_

#include "../queries.hpp"

PreparedStatement Q_19_cpar(DegreeOfParallelism dop, const aff_t& aff_parallel,
                            const aff_t& aff_reduce, DeviceType dev,
                            const scan_t& scan);

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<19>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                              Tr aff_reduce, DeviceType dev) {
  return Q_19_cpar(dop, aff_parallel, aff_reduce, dev, scan<Tplugin>);
}

#endif /* HARMONIA_QUERIES_CH_Q19_HPP_ */
