#include "experiments/realworld-queries/init.hpp"

void symantecBin1(map<string,dataset> datasetCatalog);
void symantecBin2(map<string,dataset> datasetCatalog);
void symantecBin3(map<string,dataset> datasetCatalog);


void symantecBin4(map<string,dataset> datasetCatalog);
/* XXX: Filter order matters -> Some are VERY selective */
void symantecBin4v1(map<string,dataset> datasetCatalog);

void symantecBin5(map<string,dataset> datasetCatalog);
void symantecBin6(map<string,dataset> datasetCatalog);
void symantecBin7(map<string,dataset> datasetCatalog);
void symantecBin8(map<string,dataset> datasetCatalog);

/* Less selective variations - 1 sel. predicate */
void symantecBin6v1(map<string,dataset> datasetCatalog);
void symantecBin7v1(map<string,dataset> datasetCatalog);
void symantecBin8v1(map<string,dataset> datasetCatalog);

/* Less selective variations - 2 sel. predicates - 2nd one not filtering */
void symantecBin6v2(map<string,dataset> datasetCatalog);
void symantecBin7v2(map<string,dataset> datasetCatalog);
void symantecBin8v2(map<string,dataset> datasetCatalog);
