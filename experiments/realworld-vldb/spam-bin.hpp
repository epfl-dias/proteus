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

#include "experiments/realworld-vldb/init.hpp"

/* SELECT MIN(p_event),MAX(p_event), COUNT(*) from symantecunordered where id >
 * 50000000 and id < 60000000; */
void symantecBin1(map<string, dataset> datasetCatalog);
void symantecBin2(map<string, dataset> datasetCatalog);
void symantecBin3(map<string, dataset> datasetCatalog);
void symantecBin4(map<string, dataset> datasetCatalog);
void symantecBin4v1(map<string, dataset> datasetCatalog);
void symantecBin5(map<string, dataset> datasetCatalog);
void symantecBin6(map<string, dataset> datasetCatalog);
void symantecBin6v2(map<string, dataset> datasetCatalog);
void symantecBin7(map<string, dataset> datasetCatalog);
void symantecBin7v2(map<string, dataset> datasetCatalog);
void symantecBin8(map<string, dataset> datasetCatalog);
