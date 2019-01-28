/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

#include "experiments/realworld-queries/init.hpp"

void symantecBinJSON1(map<string, dataset> datasetCatalog);
void symantecBinJSON2(map<string, dataset> datasetCatalog);
void symantecBinJSON3(map<string, dataset> datasetCatalog);
void symantecBinJSON4(map<string, dataset> datasetCatalog);
/* Slower than expected... */
void symantecBinJSON5(map<string, dataset> datasetCatalog);

/* Slower than expected... */
void symantecBinJSON3v1(map<string, dataset> datasetCatalog);
/* Splitting selection in steps */
void symantecBinJSON3v2(map<string, dataset> datasetCatalog);
/* Same functionality - split numeric select in steps */
void symantecBinJSON2v1(map<string, dataset> datasetCatalog);
/* Changed extremely permissive .bin side */
void symantecBinJSON5v1(map<string, dataset> datasetCatalog);
