#include "experiments/realworld-vldb/init.hpp"

/* Caches all numerics needed */
void symantecCSV1Caching(map<string,dataset> datasetCatalog);
/* Does not cache size */
void symantecCSV1CachingB(map<string,dataset> datasetCatalog);
void symantecCSV1(map<string,dataset> datasetCatalog);
/* Caches size */
void symantecCSV2Caching(map<string,dataset> datasetCatalog);
void symantecCSV2(map<string,dataset> datasetCatalog);
void symantecCSV3(map<string,dataset> datasetCatalog);
void symantecCSV4(map<string,dataset> datasetCatalog);
void symantecCSV5(map<string,dataset> datasetCatalog);
void symantecCSV6(map<string,dataset> datasetCatalog);
void symantecCSV7(map<string,dataset> datasetCatalog);

void symantecCSVWarmup(map<string,dataset> datasetCatalog);

/* v1: Fewer accesses to str fields */
void symantecCSV3v1(map<string,dataset> datasetCatalog);
/* Crashes */
//void symantecCSV4v1(map<string,dataset> datasetCatalog);
void symantecCSV4v2(map<string,dataset> datasetCatalog);
