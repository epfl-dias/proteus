/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef SYMANTEC_CONFIG_HPP_
#define SYMANTEC_CONFIG_HPP_

#include "common/common.hpp"
#include "values/expressionTypes.hpp"
/* Constants and macros to be used by queries targeting dataset of spam emails
 */

#ifdef LOCAL_EXEC
#define SYMANTEC_LOCAL
#endif

#ifndef SYMANTEC_LOCAL
#define SYMANTEC_SERVER
#endif

typedef struct dataset {
  string path;
  RecordType recType;
  int linehint;
} dataset;

/* Crude schema, obtained from spams100.json */
void symantecSchema(map<string, dataset> &datasetCatalog);

/* DEPRECATED */
void symantecCoreSchema(map<string, dataset> &datasetCatalog);

/* TRUNK */
void symantecCoreIDDatesSchema(map<string, dataset> &datasetCatalog);

void symantecBinSchema(map<string, dataset> &datasetCatalog);

/* Careful: Strings have no brackets! */
void symantecCSVSchema(map<string, dataset> &datasetCatalog);

#endif /* SYMANTEC_CONFIG_HPP_ */
