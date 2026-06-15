/*
-- DATE : 25/09/2023
-- FILE_PATH : benchmark/include/config/tile_based/config.hpp
-- PROJECT_NAME : fastlanes_fileformat
*/

#ifndef FASTLANES_FILEFORMAT_CONFIG_HPP
#define FASTLANES_FILEFORMAT_CONFIG_HPP

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <sys/stat.h>

namespace tile_based {
int delta = 7;
/* Used in gen.cpp */
int num_bits = 3;
/* Used in gen_d1.cpp */
int num_distinct = 28;
/* Used in gen_d2.cpp */
int mean = 4;
/* Used in gen_d3.cpp */
int alpha = 4;

struct encoded_column {
	// block_start[i] = byte at which ith block starts
	uint* block_start;
	// raw data
	uint* data;
	// number of bytes of raw data
	uint64_t data_size;
};

int index_of(std::string* arr, int len, std::string val) {
	for (int i = 0; i < len; i++) {
		if (arr[i] == val) { return i; }
	}

	return -1;
}

// 16 / 6 / 7 / 8 - not integer columns
std::string lookup(std::string col_name) {
	std::string lineorder[] = {"lo_orderkey",
	                           "lo_linenumber",
	                           "lo_custkey",
	                           "lo_partkey",
	                           "lo_suppkey",
	                           "lo_orderdate",
	                           "lo_orderpriority",
	                           "lo_shippriority",
	                           "lo_quantity",
	                           "lo_extendedprice",
	                           "lo_ordtotalprice",
	                           "lo_discount",
	                           "lo_revenue",
	                           "lo_supplycost",
	                           "lo_tax",
	                           "lo_commitdate",
	                           "lo_shipmode"};
	std::string part[]      = {
        "p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container"};
	std::string supplier[] = {"s_suppkey", "s_name", "s_address", "s_city", "s_nation", "s_region", "s_phone"};
	std::string customer[] = {
	    "c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment"};
	std::string date[] = {"d_datekey",
	                      "d_date",
	                      "d_dayofweek",
	                      "d_month",
	                      "d_year",
	                      "d_yearmonthnum",
	                      "d_yearmonth",
	                      "d_daynuminweek",
	                      "d_daynuminmonth",
	                      "d_daynuminyear",
	                      "d_sellingseason",
	                      "d_lastdayinweekfl",
	                      "d_lastdayinmonthfl",
	                      "d_holidayfl",
	                      "d_weekdayfl"};

	if (col_name[0] == 'l') {
		int index = index_of(lineorder, 17, col_name);
		return "LINEORDER" + std::to_string(index);
	} else if (col_name[0] == 's') {
		int index = index_of(supplier, 7, col_name);
		return "SUPPLIER" + std::to_string(index);
	} else if (col_name[0] == 'c') {
		int index = index_of(customer, 8, col_name);
		return "CUSTOMER" + std::to_string(index);
	} else if (col_name[0] == 'p') {
		int index = index_of(part, 9, col_name);
		return "PART" + std::to_string(index);
	} else if (col_name[0] == 'd') {
		int index = index_of(date, 15, col_name);
		return "DDATE" + std::to_string(index);
	} else if (col_name[0] == 't') {
		// test columns
		return "tile_based_" + col_name + "_";
	} else {
		std::cout << "Unknown column " << col_name << '\n';
		exit(1);
	}

	return "";
}

template <typename T>
T* loadColumn(int num_entries, std::string file_path) {
	T*            h_col = new T[num_entries];
	std::ifstream col_data(file_path.c_str(), std::ios::in | std::ios::binary);
	if (!col_data) { return NULL; }

	col_data.read((char*)h_col, num_entries * sizeof(T));
	return h_col;
}

template <typename T>
int storeColumn(int num_entries, T* h_col, std::string file_path) {
	std::ofstream col_data(file_path.c_str(), std::ios::out | std::ios::binary);
	if (!col_data) { return -1; }

	col_data.write((char*)h_col, num_entries * sizeof(T));
	return 0;
}

inline std::string get_tile_based_dir() {
	const char* data_dir = std::getenv("FLS_DATA_DIR_PATH");
	if (data_dir == nullptr) {
		data_dir = "/home/ubuntu/fff/benchmark/data/";
		// todo : remove upper line.
		//		throw std::runtime_error("FLS_DATA_DIR_PATH IS NOT SET.");
	}

	std::string result = static_cast<std::string>(data_dir) + "binary/tile_based/";

	return result;
}

inline std::string get_gen_file_path() {
	std::string col_name = std::string("gen") + "_" + std::to_string(tile_based::num_bits);
	std::string result   = get_tile_based_dir() + col_name + ".tle";

	return result;
}

inline std::string get_binpack_file_path() {
	std::string col_name = std::string("binpack") + "_" + std::to_string(tile_based::num_bits);
	std::string result   = get_tile_based_dir() + col_name + ".tle";

	return result;
}

inline std::string get_binfos_file_path() {
	std::string col_name = std::string("binofs") + "_" + std::to_string(tile_based::num_bits);
	std::string result   = get_tile_based_dir() + col_name + ".tle";

	return result;
}

/***
 * Loads encoding from disk into memory
 * encoding: bin | dbin
 **/
inline encoded_column loadEncodedColumn(int num_entries) {
	// Open file
	std::string filename         = get_binpack_file_path();
	std::string offsets_filename = get_binfos_file_path();
	int         fd               = open(filename.c_str(), O_RDONLY);

	// Get size of file
	struct stat s;
	int         status   = fstat(fd, &s);
	int         filesize = s.st_size;

	encoded_column col;

	std::ifstream col_data(filename.c_str(), std::ios::in | std::ios::binary);
	if (!col_data) {
		std::cout << "Unable to open encoded column file" << filename << std::endl;
		exit(1);
	}

	col.data = new uint[filesize / 4];
	col_data.read((char*)col.data, filesize);
	col_data.close();

	col.data_size = filesize;

	int block_size      = 128;
	int elem_per_thread = 4;
	int tile_size       = block_size * elem_per_thread;
	int adjusted_len    = ((num_entries + tile_size - 1) / tile_size) * tile_size;
	int num_blocks      = adjusted_len / block_size;

	col.block_start = new uint[num_blocks + 1];

	std::ifstream offsets_data(offsets_filename.c_str(), std::ios::in | std::ios::binary);
	if (!offsets_data) {
		std::cout << "Unable to open encoded column file" << offsets_filename << std::endl;
		exit(1);
	}

	offsets_data.read((char*)col.block_start, (num_blocks + 1) * sizeof(int));
	offsets_data.close();

	return col;
}
} // namespace tile_based
#endif // FASTLANES_FILEFORMAT_CONFIG_HPP
