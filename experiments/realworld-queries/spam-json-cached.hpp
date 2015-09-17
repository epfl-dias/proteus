#include "experiments/realworld-queries/init.hpp"

void symantecJSON1(map<string,dataset> datasetCatalog);
/* Caches a bit more than what is needed (month, year) */
void symantecJSON1Caching(map<string,dataset> datasetCatalog);
void symantecJSON2(map<string,dataset> datasetCatalog);
void symantecJSON3(map<string,dataset> datasetCatalog);
void symantecJSON4(map<string,dataset> datasetCatalog);
void symantecJSON5(map<string,dataset> datasetCatalog);
void symantecJSON6(map<string,dataset> datasetCatalog);
void symantecJSON7(map<string,dataset> datasetCatalog);
//XXX Long-running (and actually crashing with caches..?)
//void symantecJSON8(map<string,dataset> datasetCatalog);
void symantecJSON9(map<string,dataset> datasetCatalog);
void symantecJSON10(map<string,dataset> datasetCatalog);
void symantecJSON11(map<string,dataset> datasetCatalog);
void symantecJSONWarmup(map<string,dataset> datasetCatalog);
