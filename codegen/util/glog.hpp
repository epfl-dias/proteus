/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef GLOG_HPP_
#define GLOG_HPP_

// Used to remove all logging messages at compile time and not affect
// performance Must be placed before glog include
/*Setting GOOGLE_STRIP_LOG to 1 or greater removes all log messages associated
 * with VLOGs
 * as well as INFO log statements. Setting it to two removes WARNING log
 * statements too. */
#ifdef NDEBUG
#define GOOGLE_STRIP_LOG 2
#define STRIP_LOG 2
#endif

#include <glog/logging.h>

#endif /* GLOG_HPP_ */
