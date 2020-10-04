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
#include "storage/multi-version/mv.hpp"

namespace storage::mv {

// instantiate to create them
MultiVersionStorage_impl<MV_RecordList_Full> one;
MultiVersionStorage_impl<MV_RecordList_Partial> two;
MultiVersionStorage_impl<MV_perAttribute<MV_attributeList>> three;
MultiVersionStorage_impl<MV_perAttribute<MV_DAG>> four;

}  // namespace storage::mv
