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

void symantecBin1(map<string, dataset> datasetCatalog);
void symantecBin2(map<string, dataset> datasetCatalog);
void symantecBin3(map<string, dataset> datasetCatalog);

void symantecBin4(map<string, dataset> datasetCatalog);
/* XXX: Filter order matters -> Some are VERY selective */
void symantecBin4v1(map<string, dataset> datasetCatalog);

void symantecBin5(map<string, dataset> datasetCatalog);
void symantecBin6(map<string, dataset> datasetCatalog);
void symantecBin7(map<string, dataset> datasetCatalog);
void symantecBin8(map<string, dataset> datasetCatalog);

/* Less selective variations - 1 sel. predicate */
void symantecBin6v1(map<string, dataset> datasetCatalog);
void symantecBin7v1(map<string, dataset> datasetCatalog);
void symantecBin8v1(map<string, dataset> datasetCatalog);

/* Less selective variations - 2 sel. predicates - 2nd one not filtering */
void symantecBin6v2(map<string, dataset> datasetCatalog);
void symantecBin7v2(map<string, dataset> datasetCatalog);
void symantecBin8v2(map<string, dataset> datasetCatalog);
