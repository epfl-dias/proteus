#ifndef TABLE_HPP_
#define TABLE_HPP_

#include <map>
#include <vector>
#include <iostream>
#include <string>
#include <assert.h>
 
#include "storage/memory_manager.hpp"
#include "indexes/index.hpp"


namespace storage {

class Schema;
class Table;
class ColumnStore;
class RowStore;
class Row;
class Column;

class Schema {

	int num_tables;
	std::vector<Table> tables;


	std::string name;


	void create_table();
	void drop_table();
	
	Table* getTable(int idx);
	Table* getTable(std::string name);


};

class Column {
	
	
	size_t elem_size;
	std::vector<mem_chunk*> data_ptr;

	bool is_indexed;
	index::Index* index_ptr;


	void* getRange(int start_idx, int end_idx);
	void* getElem(int idx);/*{
		
		assert(data_ptr != NULL);

		int data_loc = idx * elem_size;

		for (const auto &chunk : data_ptr) {
			if(chunk->size <= (data_loc+elem_size) ){
				return chunk->data+data_loc;
			}
		}
		
	}*/

};

class Row {
	
	size_t elem_size;
	std::vector<mem_chunk*> data_ptr;

	void* getRow(int idx);
	void* getRange(int start_idx, int end_idx);

};


class Table {


public:

	virtual void deleteAllTuples() = 0;
	//virtual void deleteAllTuples() = 0;
    virtual bool insertRecord() = 0;
    virtual bool updateRecord() = 0;
    virtual bool deleteRecord() = 0;

private:
	std::string name;
	int num_columns;
	int primary_index_col_idx;

};


class rowStore : public Table {
public:

private:

};

class columnStore : public Table {
public:

	columnStore();
	
	
private:

	std::vector<Column> columns;

};

};

#endif /* TABLE_HPP_ */
