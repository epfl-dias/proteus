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

#ifndef PROTEUS_OLTP_MV_HPP
#define PROTEUS_OLTP_MV_HPP

// MV Types

#include "storage/multi-version/mv-attribute-list.hpp"
#include "storage/multi-version/mv-record-list.hpp"

namespace storage::mv {

using mv_type = MV_RecordList_Full;
// using mv_type = MV_RecordList_Partial;

//using mv_type = MV_perAttribute<MV_attributeList>;
// using mv_type = MV_perAttribute<MV_DAG>;

using mv_version_chain = mv_type::version_chain_t;
using mv_version = mv_type::version_t;

}  // namespace storage::mv

#endif  // PROTEUS_OLTP_MV_HPP
