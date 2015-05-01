/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2014
		Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef TPCH_CONFIG_HPP_
#define TPCH_CONFIG_HPP_

#include "common/common.hpp"
/* Constants and macros to be used by micro-benchmark queries */

#define TPCH_LOCAL
#ifndef TPCH_LOCAL
#define TPCH_SERVER
#define TPCH_SF10
#endif

#ifdef TPCH_SERVER
#ifdef TPCH_SF10
#define O_ORDERKEY_MIN 1
#define O_CUSTKEY_MIN 1
#define O_TOTALPRICE_MIN 838.05
#define O_SHIPPRIORITY_MIN 0

#define O_ORDERKEY_MAX 60000000
#define O_CUSTKEY_MAX 1499999
#define O_TOTALPRICE_MAX 558822.56
#define O_SHIPPRIORITY_MAX 0

#define L_ORDERKEY_MIN 1
#define L_PARTKEY_MIN 1
#define L_SUPPKEY_MIN 1
#define L_LINENUMBER_MIN 1
#define L_QUANTITY_MIN 1.0
#define L_EXTENDEDPRICE_MIN 900.91
#define L_DISCOUNT_MIN 0.00
#define L_TAX_MIN 0.00

#define L_ORDERKEY_MAX 60000000
#define L_PARTKEY_MAX 2000000
#define L_SUPPKEY_MAX 100000
#define L_LINENUMBER_MAX 7
#define L_QUANTITY_MAX 50.00
#define L_EXTENDEDPRICE_MAX 104949.50
#define L_DISCOUNT_MAX 0.10
#define L_TAX_MAX 0.08
#endif
#endif

#ifdef TPCH_LOCAL
#define O_ORDERKEY_MIN 1
#define O_CUSTKEY_MIN 369001
#define O_TOTALPRICE_MIN 41714.38
#define O_SHIPPRIORITY_MIN 0
#define O_ORDERKEY_MAX 34
#define O_CUSTKEY_MAX 1367761
#define O_TOTALPRICE_MAX 287534.80
#define O_SHIPPRIORITY_MAX 0

#define L_ORDERKEY_MIN 1
#define L_PARTKEY_MIN 21315
#define L_SUPPKEY_MIN 6348
#define L_LINENUMBER_MIN 1
#define L_QUANTITY_MIN 8.00
#define L_EXTENDEDPRICE_MIN 15479.68
#define L_DISCOUNT_MIN 0.00
#define L_TAX_MIN 0.00

#define L_ORDERKEY_MAX 3
#define L_PARTKEY_MAX 1551894
#define L_SUPPKEY_MAX 76910
#define L_LINENUMBER_MAX 6
#define L_QUANTITY_MAX 49.00
#define L_EXTENDEDPRICE_MAX 86083.65
#define L_DISCOUNT_MAX 0.10
#define L_TAX_MAX 0.07
#endif

typedef struct dataset	{
	string path;
	RecordType recType;
	int linehint;
} dataset;

void tpchSchemaCSV(map<string,dataset>& datasetCatalog)	{
	IntType *intType 		= new IntType();
	FloatType *floatType 	= new FloatType();
	StringType *stringType 	= new StringType();

	dataset lineitem;
	dataset orders;
#ifdef TPCH_LOCAL
	/* Lineitem */
	string lineitemPath = string("inputs/tpch/lineitem10.csv");
	lineitem.linehint = 10;
	/* Orders */
	string ordersPath = string("inputs/tpch/orders10.csv");
	orders.linehint = 10;
#endif
#ifdef TPCH_SF10
	/* Lineitem */
	string lineitemPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/lineitem.csv");
	lineitem.linehint = 59986052;
	/* Orders */
	string ordersPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/orders.csv");
	orders.linehint = 15000000;
#endif

	lineitem.path = lineitemPath;
	orders.path = ordersPath;

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey =
			new RecordAttribute(1, lineitemPath, "l_orderkey",intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *l_partkey =
			new RecordAttribute(2, lineitemPath, "l_partkey", intType);
	attsLineitem.push_back(l_partkey);
	RecordAttribute *l_suppkey =
			new RecordAttribute(3, lineitemPath, "l_suppkey", intType);
	attsLineitem.push_back(l_suppkey);
	RecordAttribute *l_linenumber =
			new RecordAttribute(4, lineitemPath, "l_linenumber",intType);
	attsLineitem.push_back(l_linenumber);
	RecordAttribute *l_quantity =
			new RecordAttribute(5, lineitemPath, "l_quantity", floatType);
	attsLineitem.push_back(l_quantity);
	RecordAttribute *l_extendedprice =
			new RecordAttribute(6, lineitemPath,"l_extendedprice", floatType);
	attsLineitem.push_back(l_extendedprice);
	RecordAttribute *l_discount =
			new RecordAttribute(7, lineitemPath, "l_discount",	floatType);
	attsLineitem.push_back(l_discount);
	RecordAttribute *l_tax =
			new RecordAttribute(8, lineitemPath, "l_tax", floatType);
	attsLineitem.push_back(l_tax);
	RecordAttribute *l_returnflag =
			new RecordAttribute(9, lineitemPath, "l_returnflag", stringType);
	attsLineitem.push_back(l_returnflag);
	RecordAttribute *l_linestatus =
			new RecordAttribute(10, lineitemPath, "l_linestatus", stringType);
	attsLineitem.push_back(l_linestatus);
	RecordAttribute *l_shipdate =
			new RecordAttribute(11, lineitemPath, "l_shipdate", stringType);
	attsLineitem.push_back(l_shipdate);
	RecordAttribute *l_commitdate =
			new RecordAttribute(12, lineitemPath, "l_commitdate",stringType);
	attsLineitem.push_back(l_commitdate);
	RecordAttribute *l_receiptdate =
			new RecordAttribute(13, lineitemPath, "l_receiptdate",stringType);
	attsLineitem.push_back(l_receiptdate);
	RecordAttribute *l_shipinstruct =
			new RecordAttribute(14, lineitemPath, "l_shipinstruct", stringType);
	attsLineitem.push_back(l_shipinstruct);
	RecordAttribute *l_shipmode =
			new RecordAttribute(15, lineitemPath, "l_shipmode", stringType);
	attsLineitem.push_back(l_shipmode);
	RecordAttribute *l_comment =
			new RecordAttribute(16, lineitemPath, "l_comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);
	lineitem.recType = lineitemRec;


	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey =
			new RecordAttribute(1, ordersPath, "o_orderkey",intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *o_custkey =
			new RecordAttribute(2, ordersPath, "o_custkey", intType);
	attsOrder.push_back(o_custkey);
	RecordAttribute *o_orderstatus =
			new RecordAttribute(3, ordersPath, "o_orderstatus", stringType);
	attsOrder.push_back(o_orderstatus);
	RecordAttribute *o_totalprice =
			new RecordAttribute(4, ordersPath, "o_totalprice",floatType);
	attsOrder.push_back(o_totalprice);
	RecordAttribute *o_orderdate =
			new RecordAttribute(5, ordersPath, "o_orderdate", stringType);
	attsOrder.push_back(o_orderdate);
	RecordAttribute *o_orderpriority =
			new RecordAttribute(6, ordersPath,"o_orderpriority", stringType);
	attsOrder.push_back(o_orderpriority);
	RecordAttribute *o_clerk =
			new RecordAttribute(7, ordersPath, "o_clerk",	stringType);
	attsOrder.push_back(o_clerk);
	RecordAttribute *o_shippriority =
			new RecordAttribute(8, ordersPath, "o_shippriority", intType);
	attsOrder.push_back(o_shippriority);
	RecordAttribute *o_comment =
			new RecordAttribute(9, ordersPath, "o_comment", stringType);
	attsOrder.push_back(o_comment);

	RecordType ordersRec = RecordType(attsOrder);
	orders.recType = ordersRec;


	datasetCatalog["lineitem"] = lineitem;
	datasetCatalog["orders"] 	= orders;
}

void tpchSchemaJSON(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset lineitem;
	dataset orders;
	dataset ordersLineitems;

	 #ifdef TPCH_LOCAL
	/* Lineitem */
	string lineitemPath = string("inputs/tpch/json/lineitem10.json");
//	string lineitemPath = string("inputs/tpch/json/shortest.json");
//	string lineitemPath = string("inputs/tpch/json/longest.json");
//	lineitem.linehint = 1;
	lineitem.linehint = 10;
	/* Orders */
	string ordersPath = string("inputs/tpch/json/orders10.json");
	orders.linehint = 10;
	/* OrdersLineitems
	 * i.e., pre-materialized join */
//	string ordersLineitemsPath = string(
//			"inputs/tpch/json/ordersLineitemsArray10.json");
//	ordersLineitems.linehint = 10;
	string ordersLineitemsPath = string(
			"inputs/tpch/json/ordersLineitemsArray20.json");
	ordersLineitems.linehint = 20;
	#endif
	#ifdef TPCH_SF10
	/* Lineitem */
	string lineitemPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/lineitemFlatSF10.json");
	lineitem.linehint = 59986052;
	/* Orders */
	string ordersPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/orders.json");
	orders.linehint = 15000000;
	/* OrdersLineitems
	 * i.e., pre-materialized join */
	string ordersLineitemsPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/ordersLineitemsArray.json");
	ordersLineitems.linehint = 15000000;
	#endif

	lineitem.path = lineitemPath;
	orders.path = ordersPath;
	ordersLineitems.path = ordersLineitemsPath;

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey = new RecordAttribute(1, lineitemPath,
			"orderkey", intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *partkey = new RecordAttribute(2, lineitemPath, "partkey",
			intType);
	attsLineitem.push_back(partkey);
	RecordAttribute *suppkey = new RecordAttribute(3, lineitemPath, "suppkey",
			intType);
	attsLineitem.push_back(suppkey);
	RecordAttribute *linenumber = new RecordAttribute(4, lineitemPath,
			"linenumber", intType);
	attsLineitem.push_back(linenumber);
	RecordAttribute *quantity = new RecordAttribute(5, lineitemPath, "quantity",
			floatType);
	attsLineitem.push_back(quantity);
	RecordAttribute *extendedprice = new RecordAttribute(6, lineitemPath,
			"extendedprice", floatType);
	attsLineitem.push_back(extendedprice);
	RecordAttribute *discount = new RecordAttribute(7, lineitemPath, "discount",
			floatType);
	attsLineitem.push_back(discount);
	RecordAttribute *tax = new RecordAttribute(8, lineitemPath, "tax",
			floatType);
	attsLineitem.push_back(tax);
	RecordAttribute *returnflag = new RecordAttribute(9, lineitemPath,
			"returnflag", stringType);
	attsLineitem.push_back(returnflag);
	RecordAttribute *linestatus = new RecordAttribute(10, lineitemPath,
			"linestatus", stringType);
	attsLineitem.push_back(linestatus);
	RecordAttribute *shipdate = new RecordAttribute(11, lineitemPath,
			"shipdate", stringType);
	attsLineitem.push_back(shipdate);
	RecordAttribute *commitdate = new RecordAttribute(12, lineitemPath,
			"commitdate", stringType);
	attsLineitem.push_back(commitdate);
	RecordAttribute *receiptdate = new RecordAttribute(13, lineitemPath,
			"receiptdate", stringType);
	attsLineitem.push_back(receiptdate);
	RecordAttribute *shipinstruct = new RecordAttribute(14, lineitemPath,
			"shipinstruct", stringType);
	attsLineitem.push_back(shipinstruct);
	RecordAttribute *shipmode = new RecordAttribute(15, lineitemPath,
			"shipmode", stringType);
	attsLineitem.push_back(shipmode);
	RecordAttribute *l_comment = new RecordAttribute(16, lineitemPath,
			"comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);
	lineitem.recType = lineitemRec;

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey = new RecordAttribute(1, ordersPath, "orderkey",
			intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *custkey = new RecordAttribute(2, ordersPath, "custkey",
			intType);
	attsOrder.push_back(custkey);
	RecordAttribute *orderstatus = new RecordAttribute(3, ordersPath,
			"orderstatus", stringType);
	attsOrder.push_back(orderstatus);
	RecordAttribute *totalprice = new RecordAttribute(4, ordersPath,
			"totalprice", floatType);
	attsOrder.push_back(totalprice);
	RecordAttribute *orderdate = new RecordAttribute(5, ordersPath, "orderdate",
			stringType);
	attsOrder.push_back(orderdate);
	RecordAttribute *orderpriority = new RecordAttribute(6, ordersPath,
			"orderpriority", stringType);
	attsOrder.push_back(orderpriority);
	RecordAttribute *clerk = new RecordAttribute(7, ordersPath, "clerk",
			stringType);
	attsOrder.push_back(clerk);
	RecordAttribute *shippriority = new RecordAttribute(8, ordersPath,
			"shippriority", intType);
	attsOrder.push_back(shippriority);
	RecordAttribute *o_comment = new RecordAttribute(9, ordersPath, "comment",
			stringType);
	attsOrder.push_back(o_comment);

	RecordType ordersRec = RecordType(attsOrder);
	orders.recType = ordersRec;

	/* XXX ERROR PRONE...
	 * outer and nested attributes have the same relationName
	 * Don't know how caches will work in this scenario.
	 *
	 * The rest of the code should be sanitized,
	 * since readPath is applied in steps.
	 */

	/*
	 * The lineitem entries in the ordersLineitems objects
	 * do not contain the orderkey (again)
	 */
	list<RecordAttribute*> attsLineitemNested = list<RecordAttribute*>();
	{
		RecordAttribute *partkey = new RecordAttribute(1, ordersLineitemsPath,
				"partkey", intType);
		attsLineitemNested.push_back(partkey);
		RecordAttribute *suppkey = new RecordAttribute(2, ordersLineitemsPath,
				"suppkey", intType);
		attsLineitemNested.push_back(suppkey);
		RecordAttribute *linenumber = new RecordAttribute(3,
				ordersLineitemsPath, "linenumber", intType);
		attsLineitemNested.push_back(linenumber);
		RecordAttribute *quantity = new RecordAttribute(4, ordersLineitemsPath,
				"quantity", floatType);
		attsLineitemNested.push_back(quantity);
		RecordAttribute *extendedprice = new RecordAttribute(5,
				ordersLineitemsPath, "extendedprice", floatType);
		attsLineitemNested.push_back(extendedprice);
		RecordAttribute *discount = new RecordAttribute(6, ordersLineitemsPath,
				"discount", floatType);
		attsLineitemNested.push_back(discount);
		RecordAttribute *tax = new RecordAttribute(7, ordersLineitemsPath,
				"tax", floatType);
		attsLineitemNested.push_back(tax);
		RecordAttribute *returnflag = new RecordAttribute(8,
				ordersLineitemsPath, "returnflag", stringType);
		attsLineitemNested.push_back(returnflag);
		RecordAttribute *linestatus = new RecordAttribute(9,
				ordersLineitemsPath, "linestatus", stringType);
		attsLineitemNested.push_back(linestatus);
		RecordAttribute *shipdate = new RecordAttribute(10, ordersLineitemsPath,
				"shipdate", stringType);
		attsLineitemNested.push_back(shipdate);
		RecordAttribute *commitdate = new RecordAttribute(11,
				ordersLineitemsPath, "commitdate", stringType);
		attsLineitemNested.push_back(commitdate);
		RecordAttribute *receiptdate = new RecordAttribute(12,
				ordersLineitemsPath, "receiptdate", stringType);
		attsLineitemNested.push_back(receiptdate);
		RecordAttribute *shipinstruct = new RecordAttribute(13,
				ordersLineitemsPath, "shipinstruct", stringType);
		attsLineitemNested.push_back(shipinstruct);
		RecordAttribute *shipmode = new RecordAttribute(14, ordersLineitemsPath,
				"shipmode", stringType);
		attsLineitemNested.push_back(shipmode);
		RecordAttribute *l_comment = new RecordAttribute(15,
				ordersLineitemsPath, "comment", stringType);
		attsLineitemNested.push_back(l_comment);
	}
	RecordType *lineitemNestedRec = new RecordType(attsLineitemNested);

	list<RecordAttribute*> attsOrdersLineitems = list<RecordAttribute*>();
	{
		RecordAttribute *o_orderkey = new RecordAttribute(1,
				ordersLineitemsPath, "orderkey", intType);
		attsOrdersLineitems.push_back(o_orderkey);
		RecordAttribute *custkey = new RecordAttribute(2, ordersLineitemsPath,
				"custkey", intType);
		attsOrdersLineitems.push_back(custkey);
		RecordAttribute *orderstatus = new RecordAttribute(3,
				ordersLineitemsPath, "orderstatus", stringType);
		attsOrdersLineitems.push_back(orderstatus);
		RecordAttribute *totalprice = new RecordAttribute(4,
				ordersLineitemsPath, "totalprice", floatType);
		attsOrdersLineitems.push_back(totalprice);
		RecordAttribute *orderdate = new RecordAttribute(5, ordersLineitemsPath,
				"orderdate", stringType);
		attsOrdersLineitems.push_back(orderdate);
		RecordAttribute *orderpriority = new RecordAttribute(6,
				ordersLineitemsPath, "orderpriority", stringType);
		attsOrdersLineitems.push_back(orderpriority);
		RecordAttribute *clerk = new RecordAttribute(7, ordersLineitemsPath,
				"clerk", stringType);
		attsOrdersLineitems.push_back(clerk);
		RecordAttribute *shippriority = new RecordAttribute(8,
				ordersLineitemsPath, "shippriority", intType);
		attsOrdersLineitems.push_back(shippriority);
		RecordAttribute *o_comment = new RecordAttribute(9, ordersLineitemsPath,
				"comment", stringType);
		attsOrdersLineitems.push_back(o_comment);
		RecordAttribute *lineitems = new RecordAttribute(10,
				ordersLineitemsPath, "lineitems", lineitemNestedRec);
		attsOrdersLineitems.push_back(lineitems);
	}
	RecordType ordersLineitemsRec = RecordType(attsOrdersLineitems);
	ordersLineitems.recType = ordersLineitemsRec;

	datasetCatalog["lineitem"] = lineitem;
	datasetCatalog["orders"] = orders;
	datasetCatalog["ordersLineitems"] = ordersLineitems;
}

void tpchSchemaBin(map<string,dataset>& datasetCatalog)	{
	IntType *intType 		= new IntType();
	FloatType *floatType 	= new FloatType();
	StringType *stringType 	= new StringType();

	dataset lineitem;
	dataset orders;
#ifdef TPCH_LOCAL
	/* Lineitem */
	string lineitemPath = string("inputs/tpch/col10/lineitem");
	lineitem.linehint = 10;
	/* Orders */
	string ordersPath = string("inputs/tpch/col10/orders");
	orders.linehint = 10;
#endif
#ifdef TPCH_SF10
	/* Lineitem */
	string lineitemPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/col/lineitem");
	lineitem.linehint = 59986052;
	/* Orders */
	string ordersPath = string("/cloud_store/manosk/data/vida-engine/tpch_2_17_0/sf10/col/orders");
	orders.linehint = 15000000;
#endif

	lineitem.path = lineitemPath;
	orders.path = ordersPath;

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey =
			new RecordAttribute(1, lineitemPath, "l_orderkey",intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *l_partkey =
			new RecordAttribute(2, lineitemPath, "l_partkey", intType);
	attsLineitem.push_back(l_partkey);
	RecordAttribute *l_suppkey =
			new RecordAttribute(3, lineitemPath, "l_suppkey", intType);
	attsLineitem.push_back(l_suppkey);
	RecordAttribute *l_linenumber =
			new RecordAttribute(4, lineitemPath, "l_linenumber",intType);
	attsLineitem.push_back(l_linenumber);
	RecordAttribute *l_quantity =
			new RecordAttribute(5, lineitemPath, "l_quantity", floatType);
	attsLineitem.push_back(l_quantity);
	RecordAttribute *l_extendedprice =
			new RecordAttribute(6, lineitemPath,"l_extendedprice", floatType);
	attsLineitem.push_back(l_extendedprice);
	RecordAttribute *l_discount =
			new RecordAttribute(7, lineitemPath, "l_discount",	floatType);
	attsLineitem.push_back(l_discount);
	RecordAttribute *l_tax =
			new RecordAttribute(8, lineitemPath, "l_tax", floatType);
	attsLineitem.push_back(l_tax);
	RecordAttribute *l_returnflag =
			new RecordAttribute(9, lineitemPath, "l_returnflag", stringType);
	attsLineitem.push_back(l_returnflag);
	RecordAttribute *l_linestatus =
			new RecordAttribute(10, lineitemPath, "l_linestatus", stringType);
	attsLineitem.push_back(l_linestatus);
	RecordAttribute *l_shipdate =
			new RecordAttribute(11, lineitemPath, "l_shipdate", stringType);
	attsLineitem.push_back(l_shipdate);
	RecordAttribute *l_commitdate =
			new RecordAttribute(12, lineitemPath, "l_commitdate",stringType);
	attsLineitem.push_back(l_commitdate);
	RecordAttribute *l_receiptdate =
			new RecordAttribute(13, lineitemPath, "l_receiptdate",stringType);
	attsLineitem.push_back(l_receiptdate);
	RecordAttribute *l_shipinstruct =
			new RecordAttribute(14, lineitemPath, "l_shipinstruct", stringType);
	attsLineitem.push_back(l_shipinstruct);
	RecordAttribute *l_shipmode =
			new RecordAttribute(15, lineitemPath, "l_shipmode", stringType);
	attsLineitem.push_back(l_shipmode);
	RecordAttribute *l_comment =
			new RecordAttribute(16, lineitemPath, "l_comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);
	lineitem.recType = lineitemRec;

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey =
			new RecordAttribute(1, ordersPath, "o_orderkey",intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *o_custkey =
			new RecordAttribute(2, ordersPath, "o_custkey", intType);
	attsOrder.push_back(o_custkey);
	RecordAttribute *o_orderstatus =
			new RecordAttribute(3, ordersPath, "o_orderstatus", stringType);
	attsOrder.push_back(o_orderstatus);
	RecordAttribute *o_totalprice =
			new RecordAttribute(4, ordersPath, "o_totalprice",floatType);
	attsOrder.push_back(o_totalprice);
	RecordAttribute *o_orderdate =
			new RecordAttribute(5, ordersPath, "o_orderdate", stringType);
	attsOrder.push_back(o_orderdate);
	RecordAttribute *o_orderpriority =
			new RecordAttribute(6, ordersPath,"o_orderpriority", stringType);
	attsOrder.push_back(o_orderpriority);
	RecordAttribute *o_clerk =
			new RecordAttribute(7, ordersPath, "o_clerk",	stringType);
	attsOrder.push_back(o_clerk);
	RecordAttribute *o_shippriority =
			new RecordAttribute(8, ordersPath, "o_shippriority", intType);
	attsOrder.push_back(o_shippriority);
	RecordAttribute *o_comment =
			new RecordAttribute(9, ordersPath, "o_comment", stringType);
	attsOrder.push_back(o_comment);

	RecordType ordersRec = RecordType(attsOrder);
	orders.recType = ordersRec;

	datasetCatalog["lineitem"] = lineitem;
	datasetCatalog["orders"] 	= orders;
}


#endif /* TPCH_CONFIG_HPP_ */
