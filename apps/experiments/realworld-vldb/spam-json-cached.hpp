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

#include "experiments/realworld-vldb/init.hpp"

void symantecJSON1(map<string, dataset> datasetCatalog);
/* Caches a bit more than what is needed (month, year) */
void symantecJSON1Caching(map<string, dataset> datasetCatalog);
void symantecJSON2(map<string, dataset> datasetCatalog);
void symantecJSON3(map<string, dataset> datasetCatalog);
void symantecJSON4(map<string, dataset> datasetCatalog);
void symantecJSON5(map<string, dataset> datasetCatalog);
void symantecJSON6(map<string, dataset> datasetCatalog);
void symantecJSON7(map<string, dataset> datasetCatalog);
// XXX Long-running (and actually crashing with caches..?)
// void symantecJSON8(map<string,dataset> datasetCatalog);
void symantecJSON9(map<string, dataset> datasetCatalog);
void symantecJSON10(map<string, dataset> datasetCatalog);
void symantecJSON11(map<string, dataset> datasetCatalog);
void symantecJSONWarmup(map<string, dataset> datasetCatalog);
/* Changed 2v1 again for optimization reasons: Step-wise select ops. */
void symantecJSON2v1(map<string, dataset> datasetCatalog);
void symantecJSON3v1(map<string, dataset> datasetCatalog);
void symantecJSON4v1(map<string, dataset> datasetCatalog);
void symantecJSON5v1(map<string, dataset> datasetCatalog);
void symantecJSON6v1(map<string, dataset> datasetCatalog);
void symantecJSON7v1(map<string, dataset> datasetCatalog);
void symantecJSON9v1(map<string, dataset> datasetCatalog);
void symantecJSON10v1(map<string, dataset> datasetCatalog);
void symantecJSON11v1(map<string, dataset> datasetCatalog);
