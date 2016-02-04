#include "experiments/realworld-vldb/spam-bin.hpp"
#include "experiments/realworld-vldb/spam-csv-cached-columnar.hpp"
//#include "experiments/realworld-vldb/spam-json-cached.hpp"
#include "experiments/realworld-vldb/spam-bin-csv.hpp"
//#include "experiments/realworld-vldb/spam-bin-json.hpp"
//#include "experiments/realworld-vldb/spam-csv-json.hpp"
//#include "experiments/realworld-vldb/spam-bin-csv-json-v2.hpp"



int main()	{
	cout << "[ViDa] Execution - Entire Workload" << endl;
	map<string,dataset> datasetCatalog;
	symantecBinSchema(datasetCatalog);
	symantecCSVSchema(datasetCatalog);
	symantecCoreIDDatesSchema(datasetCatalog);


//	cout << "SYMANTEC BIN 1" << endl;
//	symantecBin1(datasetCatalog);
//	cout << "SYMANTEC BIN 2" << endl;
//	symantecBin2(datasetCatalog);
//	cout << "SYMANTEC BIN 3" << endl;
//	symantecBin3(datasetCatalog);
//	cout << "SYMANTEC BIN 4v1" << endl;
//	symantecBin4v1(datasetCatalog);
//	cout << "SYMANTEC BIN 5" << endl;
//	symantecBin5(datasetCatalog);
//	cout << "SYMANTEC BIN 6" << endl;
//	symantecBin6v2(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC BIN 7" << endl;
//	symantecBin7v2(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC BIN 8" << endl;
//	symantecBin8v2(datasetCatalog);
//	cout << "**************" << endl;


	cout << "SYMANTEC CSV 1 (+Caching)" << endl;
	symantecCSV1Caching(datasetCatalog);
	cout << "**************" << endl;
	cout << "SYMANTEC CSV 2 (+Caching)" << endl;
	symantecCSV2Caching(datasetCatalog);
	cout << "**************" << endl;
//	cout << "SYMANTEC CSV 3v1" << endl;
//	symantecCSV3v1(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC CSV 4v1" << endl;
//	symantecCSV4v2(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC CSV 5" << endl;
//	symantecCSV5(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC CSV 6" << endl;
//	symantecCSV6(datasetCatalog);
//	cout << "**************" << endl;
//	cout << "SYMANTEC CSV 7" << endl;
//	symantecCSV7(datasetCatalog);
//	cout << "**************" << endl;


//	cout << "SYMANTEC JSON 1 (PM+Very Eager Caching)" << endl;
//	symantecJSON1Caching(datasetCatalog);
//	cout << "SYMANTEC JSON 2" << endl;
//	symantecJSON2v1(datasetCatalog);
//	cout << "SYMANTEC JSON 3" << endl;
//	symantecJSON3v1(datasetCatalog);
//	cout << "SYMANTEC JSON 4" << endl;
//	symantecJSON4v1(datasetCatalog);
//	cout << "SYMANTEC JSON 5" << endl;
//	symantecJSON5v1(datasetCatalog);
//	cout << "SYMANTEC JSON 6" << endl;
//	symantecJSON6v1(datasetCatalog);
//	cout << "SYMANTEC JSON 7" << endl;
//	symantecJSON7v1(datasetCatalog);
//	cout << "SYMANTEC JSON 9" << endl;
//	symantecJSON9v1(datasetCatalog);
//	cout << "SYMANTEC JSON 10" << endl;
//	symantecJSON10v1(datasetCatalog);
//	cout << "SYMANTEC JSON 11" << endl;
//	symantecJSON11v1(datasetCatalog);


	cout << "SYMANTEC BIN-CSV 1" << endl;
	symantecBinCSV1(datasetCatalog);
	cout << "SYMANTEC BIN-CSV 2" << endl;
	symantecBinCSV2(datasetCatalog);
	cout << "SYMANTEC BIN-CSV 3" << endl;
	symantecBinCSV3(datasetCatalog);
	cout << "SYMANTEC BIN-CSV 4" << endl;
	symantecBinCSV4(datasetCatalog);
	cout << "SYMANTEC BIN-CSV 5" << endl;
	symantecBinCSV5(datasetCatalog);

//
//	cout << "SYMANTEC BIN-JSON 1" << endl;
//	symantecBinJSON1(datasetCatalog);
//	cout << "SYMANTEC BIN-JSON 2" << endl;
//	symantecBinJSON2(datasetCatalog);
//	cout << "SYMANTEC BIN-JSON 3v2" << endl;
//	symantecBinJSON3v2(datasetCatalog);
//	cout << "SYMANTEC BIN-JSON 4" << endl;
//	symantecBinJSON4(datasetCatalog);
//	cout << "SYMANTEC BIN-JSON 5v1" << endl;
//	symantecBinJSON5v1(datasetCatalog);
//
//	cout << "SYMANTEC CSV-JSON 1" << endl;
//	symantecCSVJSON1(datasetCatalog);
//	cout << "SYMANTEC CSV-JSON 2" << endl;
//	symantecCSVJSON2(datasetCatalog);
//	cout << "SYMANTEC CSV-JSON 3" << endl;
//	symantecCSVJSON3(datasetCatalog);
//	cout << "SYMANTEC CSV-JSON 4" << endl;
//	symantecCSVJSON4(datasetCatalog);
//	cout << "SYMANTEC CSV-JSON 5" << endl;
//	symantecCSVJSON5v1(datasetCatalog);
//
//
//	cout << "SYMANTEC BIN-CSV-JSON 1" << endl;
//	symantecBinCSVJSON1(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 2v1" << endl;
//	symantecBinCSVJSON2v1(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 3" << endl;
//	symantecBinCSVJSON3(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 4" << endl;
//	symantecBinCSVJSON4(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 5" << endl;
//	symantecBinCSVJSON5(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 6" << endl;
//	symantecBinCSVJSON6(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 7" << endl;
//	symantecBinCSVJSON7v1(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 8" << endl;
//	symantecBinCSVJSON8v1(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 9" << endl;
//	symantecBinCSVJSON9(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 10" << endl;
//	symantecBinCSVJSON10(datasetCatalog);
}

