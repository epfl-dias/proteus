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

#include "experiments/realworld-queries/init.hpp"

/* Caches all numerics needed */
void symantecCSV1Caching(map<string, dataset> datasetCatalog);
/* Does not cache size */
void symantecCSV1CachingB(map<string, dataset> datasetCatalog);
void symantecCSV1(map<string, dataset> datasetCatalog);
/* Caches size */
void symantecCSV2Caching(map<string, dataset> datasetCatalog);
void symantecCSV2(map<string, dataset> datasetCatalog);
void symantecCSV3(map<string, dataset> datasetCatalog);
void symantecCSV4(map<string, dataset> datasetCatalog);
void symantecCSV5(map<string, dataset> datasetCatalog);
void symantecCSV6(map<string, dataset> datasetCatalog);
void symantecCSV7(map<string, dataset> datasetCatalog);

void symantecCSVWarmup(map<string, dataset> datasetCatalog);

/* v1: Fewer accesses to str fields */
void symantecCSV3v1(map<string, dataset> datasetCatalog);
/* Crashes */
// void symantecCSV4v1(map<string,dataset> datasetCatalog);
void symantecCSV4v2(map<string, dataset> datasetCatalog);
