#include "experiments/realworld-queries/init.hpp"

void symantecBinJSON1(map<string,dataset> datasetCatalog);
void symantecBinJSON2(map<string,dataset> datasetCatalog);
void symantecBinJSON3(map<string,dataset> datasetCatalog);
void symantecBinJSON4(map<string,dataset> datasetCatalog);
/* Slower than expected... */
void symantecBinJSON5(map<string,dataset> datasetCatalog);

/* Slower than expected... */
void symantecBinJSON3v1(map<string,dataset> datasetCatalog);
/* Splitting selection in steps */
void symantecBinJSON3v2(map<string,dataset> datasetCatalog);
/* Same functionality - split numeric select in steps */
void symantecBinJSON2v1(map<string,dataset> datasetCatalog);
/* Changed extremely permissive .bin side */
void symantecBinJSON5v1(map<string,dataset> datasetCatalog);
