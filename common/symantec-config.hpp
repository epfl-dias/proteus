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

#ifndef SYMANTEC_CONFIG_HPP_
#define SYMANTEC_CONFIG_HPP_

#include "common/common.hpp"
/* Constants and macros to be used by queries targeting dataset of spam emails */

//#define SYMANTEC_LOCAL
#ifndef SYMANTEC_LOCAL
#define SYMANTEC_SERVER
#endif

typedef struct dataset	{
	string path;
	RecordType recType;
	int linehint;
} dataset;


/* Crude schema, obtained from spams100.json */
void symantecSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset symantec;

	#ifdef SYMANTEC_LOCAL
//	string path = string("inputs/json/spam/spams100k.json");
//	symantec.linehint = 100000;

	string path = string("inputs/json/spam/spams1000.json");
	symantec.linehint = 1000;

//	string path = string("inputs/json/spam/outliers.json");
//	symantec.linehint = 2;
	#endif
	#ifdef SYMANTEC_SERVER

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spams56m.json");
//	symantec.linehint = 55833807;

	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spams1m.json");
	symantec.linehint = 999582;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spams10m.json");
//	symantec.linehint = 9995329;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spams100k.json");
//	symantec.linehint = 99967;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spams_head.json");
//	symantec.linehint = 100000;
	#endif
	symantec.path = path;

	list<RecordAttribute*> attsSymantec = list<RecordAttribute*>();

	/* "IP" : "83.149.45.128" */
	RecordAttribute *ip = new RecordAttribute(1, path, "IP", stringType);
	attsSymantec.push_back(ip);
	/* "_id" : { "$oid" : "4ebbd37d466e8b0b55000000" } */
	RecordAttribute *oid = new RecordAttribute(1, path, "$oid", stringType);
	list<RecordAttribute*> oidNested = list<RecordAttribute*>();
	oidNested.push_back(oid);
	RecordType *idRec = new RecordType(oidNested);
	RecordAttribute *_id = new RecordAttribute(2, path, "_id", idRec);
	attsSymantec.push_back(_id);
	/* "attach" : [ "whigmaleerie.jpg" ] (but tends to be empty) */
	ListType *attachList = new ListType(*stringType);
	RecordAttribute *attach = new RecordAttribute(3, path, "attach",
			attachList);
	attsSymantec.push_back(attach);
	/* "body_txt_a" : "blablabla" */
	RecordAttribute *body_txt_a = new RecordAttribute(4, path, "body_txt_a",
			stringType);
	attsSymantec.push_back(body_txt_a);
	/* "bot" : "Unclassified" */
	RecordAttribute *bot = new RecordAttribute(5, path, "bot", stringType);
	attsSymantec.push_back(bot);
	/* "charset" : "windows-1252" */
	RecordAttribute *charset = new RecordAttribute(6, path, "charset",
			stringType);
	attsSymantec.push_back(charset);
	/* "city" : "Ryazan" */
	RecordAttribute *city = new RecordAttribute(7, path, "city", stringType);
	attsSymantec.push_back(city);
	/* "classA" : "83" */
	RecordAttribute *classA = new RecordAttribute(8, path, "classA",
			stringType);
	attsSymantec.push_back(classA);
	/* "classB" : "83.149" */
	RecordAttribute *classB = new RecordAttribute(9, path, "classB",
			stringType);
	attsSymantec.push_back(classB);
	/* "classC" : "83.149" */
	RecordAttribute *classC = new RecordAttribute(10, path, "classC",
			stringType);
	attsSymantec.push_back(classC);
	/* "content_type" : [ "text/html", "text/plain" ] */
	ListType *contentList = new ListType(*stringType);
	RecordAttribute *content_type = new RecordAttribute(11, path,
			"content_type", contentList);
	attsSymantec.push_back(content_type);
	/* "country" : "Russian Federation" */
	RecordAttribute *country = new RecordAttribute(12, path, "country",
			stringType);
	attsSymantec.push_back(country);
	/* "country_code" : "RU" */
	RecordAttribute *country_code = new RecordAttribute(13, path,
			"country_code", stringType);
	attsSymantec.push_back(country_code);
	/* "cte" : "unknown" */
	RecordAttribute *cte = new RecordAttribute(14, path, "cte", stringType);
	attsSymantec.push_back(cte);
	/* "date" : { "$date" : 1285919417000 } */
	RecordAttribute *date_ = new RecordAttribute(1, path, "$date", floatType);
	list<RecordAttribute*> dateNested = list<RecordAttribute*>();
	dateNested.push_back(date_);
	RecordType *dateRec = new RecordType(dateNested);
	RecordAttribute *date = new RecordAttribute(15, path, "date", dateRec);
	attsSymantec.push_back(date);
	/* "day" : "2010-10-01" */
	RecordAttribute *day = new RecordAttribute(16, path, "day", stringType);
	attsSymantec.push_back(day);
	/* "from_domain" : "domain733674.com" */
	RecordAttribute *from_domain = new RecordAttribute(17, path, "from_domain",
			stringType);
	attsSymantec.push_back(from_domain);
	/* "host" : "airtelbroadband.in (but tends to be empty) */
	RecordAttribute *host = new RecordAttribute(18, path, "host", stringType);
	attsSymantec.push_back(host);
	/* "lang" : "english" */
	RecordAttribute *lang = new RecordAttribute(19, path, "lang", stringType);
	attsSymantec.push_back(lang);
	/* "lat" : 54.6197 */
	RecordAttribute *lat = new RecordAttribute(20, path, "lat", floatType);
	attsSymantec.push_back(lat);
	/* "long" : 39.74 */
	RecordAttribute *long_ = new RecordAttribute(21, path, "long", floatType);
	attsSymantec.push_back(long_);
	/* "rcpt_domain" : "domain555065.com" */
	RecordAttribute *rcpt_domain = new RecordAttribute(22, path, "rcpt_domain",
			stringType);
	attsSymantec.push_back(rcpt_domain);
	/* "size" : 3712 */
	RecordAttribute *size = new RecordAttribute(23, path, "size", intType);
	attsSymantec.push_back(size);
	/* "subject" : "LinkedIn Messages, 9/30/2010" */
	RecordAttribute *subject = new RecordAttribute(24, path, "subject",
			stringType);
	attsSymantec.push_back(subject);
	/* "uri" : [ "http://hetfonteintje.com/1.html" ] */
	ListType *uriList = new ListType(*stringType);
	RecordAttribute *uri = new RecordAttribute(25, path, "uri", uriList);
	attsSymantec.push_back(uri);
	/* "uri_domain" : [ "hetfonteintje.com" ] */
	ListType *domainList = new ListType(*stringType);
	RecordAttribute *domain = new RecordAttribute(26, path, "uriDomain",
			domainList);
	attsSymantec.push_back(domain);
	/* "uri_tld" : [ ".com" ] */
	ListType *tldList = new ListType(*stringType);
	RecordAttribute *uri_tld = new RecordAttribute(27, path, "uri_tld",
			tldList);
	attsSymantec.push_back(uri_tld);
	/* "x_p0f_detail" : "XP/2000" */
	RecordAttribute *x_p0f_detail = new RecordAttribute(28, path,
			"x_p0f_detail", stringType);
	attsSymantec.push_back(x_p0f_detail);
	/* "x_p0f_genre" : "Windows" */
	RecordAttribute *x_p0f_genre = new RecordAttribute(29, path, "x_p0f_genre",
			stringType);
	attsSymantec.push_back(x_p0f_genre);

	/* "x_p0f_signature" : "64380:116:1:48:M1460,N,N,S:." */
	RecordAttribute *x_p0f_signature = new RecordAttribute(29, path,
			"x_p0f_signature", stringType);
	attsSymantec.push_back(x_p0f_signature);

	RecordType symantecRec = RecordType(attsSymantec);
	symantec.recType = symantecRec;

	datasetCatalog["symantec"] = symantec;
}

void symantecCoreSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset symantec;

	#ifdef SYMANTEC_LOCAL
	string path = string("inputs/json/spam/spamsCoreID100.json");
	symantec.linehint = 100;

//	string path = string("inputs/json/spam/spamsCore1m.json");
//	symantec.linehint = 1000000;
	#endif
	#ifdef SYMANTEC_SERVER
//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreID28m.json");
//	symantec.linehint = 28000000;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreID1m.json");
//	symantec.linehint = 1000000;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreID28m.json");
//	//symantec.linehint = 27991113; //no-sanitize
//	symantec.linehint = 23806486; //after-sanitize

	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreID28m.json");
	symantec.linehint = 27991116;

//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreID100.json");
//	symantec.linehint = 100;
	#endif
	symantec.path = path;

	list<RecordAttribute*> attsSymantec = list<RecordAttribute*>();

	int attrCnt = 1;
	RecordAttribute *id = new RecordAttribute(attrCnt, path, "id", intType);
	attsSymantec.push_back(id);
	attrCnt++;
	/* "IP" : "83.149.45.128" */
	RecordAttribute *ip = new RecordAttribute(attrCnt, path, "IP", stringType);
	attsSymantec.push_back(ip);
	attrCnt++;
	/* "_id" : { "$oid" : "4ebbd37d466e8b0b55000000" } */
	RecordAttribute *oid = new RecordAttribute(1, path, "$oid", stringType);
	list<RecordAttribute*> oidNested = list<RecordAttribute*>();
	oidNested.push_back(oid);
	RecordType *idRec = new RecordType(oidNested);
	RecordAttribute *_id = new RecordAttribute(attrCnt, path, "_id", idRec);
	attsSymantec.push_back(_id);
	attrCnt++;
	/* "attach" : [ "whigmaleerie.jpg" ] (but tends to be empty) */
	ListType *attachList = new ListType(*stringType);
	RecordAttribute *attach = new RecordAttribute(attrCnt, path, "attach",
			attachList);
	attsSymantec.push_back(attach);
	attrCnt++;
	/* "body_txt_a" : "blablabla" */
	RecordAttribute *body_txt_a = new RecordAttribute(attrCnt, path, "body_txt_a",
			stringType);
	attsSymantec.push_back(body_txt_a);
	attrCnt++;
	/* "charset" : "windows-1252" */
	RecordAttribute *charset = new RecordAttribute(attrCnt, path, "charset",
			stringType);
	attsSymantec.push_back(charset);
	attrCnt++;
	/* "city" : "Ryazan" */
	RecordAttribute *city = new RecordAttribute(attrCnt, path, "city", stringType);
	attsSymantec.push_back(city);
	attrCnt++;
	/* "content_type" : [ "text/html", "text/plain" ] */
	ListType *contentList = new ListType(*stringType);
	RecordAttribute *content_type = new RecordAttribute(attrCnt, path,
			"content_type", contentList);
	attsSymantec.push_back(content_type);
	attrCnt++;
	/* "country_code" : "RU" */
	RecordAttribute *country_code = new RecordAttribute(attrCnt, path,
			"country_code", stringType);
	attsSymantec.push_back(country_code);
	attrCnt++;
	/* "cte" : "unknown" */
	RecordAttribute *cte = new RecordAttribute(attrCnt, path, "cte", stringType);
	attsSymantec.push_back(cte);
	/* "date" : { "$date" : 1285919417000 } */
	RecordAttribute *date_ = new RecordAttribute(1, path, "$date", floatType);
	list<RecordAttribute*> dateNested = list<RecordAttribute*>();
	dateNested.push_back(date_);
	RecordType *dateRec = new RecordType(dateNested);
	RecordAttribute *date = new RecordAttribute(attrCnt, path, "date", dateRec);
	attsSymantec.push_back(date);
	attrCnt++;
	/* "day" : "2010-10-01" */
	RecordAttribute *day = new RecordAttribute(attrCnt, path, "day", stringType);
	attsSymantec.push_back(day);
	attrCnt++;
	/* "from_domain" : "domain733674.com" */
	RecordAttribute *from_domain = new RecordAttribute(attrCnt, path, "from_domain",
			stringType);
	attsSymantec.push_back(from_domain);
	attrCnt++;
	/* "host" : "airtelbroadband.in (but tends to be empty) */
	RecordAttribute *host = new RecordAttribute(attrCnt, path, "host", stringType);
	attsSymantec.push_back(host);
	attrCnt++;
	/* "lang" : "english" */
	RecordAttribute *lang = new RecordAttribute(attrCnt, path, "lang", stringType);
	attsSymantec.push_back(lang);
	attrCnt++;
	/* "lat" : 54.6197 */
	RecordAttribute *lat = new RecordAttribute(attrCnt, path, "lat", floatType);
	attsSymantec.push_back(lat);
	attrCnt++;
	/* "long" : 39.74 */
	RecordAttribute *long_ = new RecordAttribute(attrCnt, path, "long", floatType);
	attsSymantec.push_back(long_);
	attrCnt++;
	/* "rcpt_domain" : "domain555065.com" */
	RecordAttribute *rcpt_domain = new RecordAttribute(attrCnt, path, "rcpt_domain",
			stringType);
	attsSymantec.push_back(rcpt_domain);
	attrCnt++;
	/* "size" : 3712 */
	RecordAttribute *size = new RecordAttribute(attrCnt, path, "size", intType);
	attsSymantec.push_back(size);
	attrCnt++;
	/* "subject" : "LinkedIn Messages, 9/30/2010" */
	RecordAttribute *subject = new RecordAttribute(attrCnt, path, "subject",
			stringType);
	attsSymantec.push_back(subject);
	attrCnt++;
	/* "uri" : [ "http://hetfonteintje.com/1.html" ] */
	ListType *uriList = new ListType(*stringType);
	RecordAttribute *uri = new RecordAttribute(attrCnt, path, "uri", uriList);
	attsSymantec.push_back(uri);
	attrCnt++;
	/* "uri_domain" : [ "hetfonteintje.com" ] */
	ListType *domainList = new ListType(*stringType);
	RecordAttribute *domain = new RecordAttribute(attrCnt, path, "uri_domain",
			domainList);
	attsSymantec.push_back(domain);
	/* "uri_tld" : [ ".com" ] */
	ListType *tldList = new ListType(*stringType);
	RecordAttribute *uri_tld = new RecordAttribute(attrCnt, path, "uri_tld",
			tldList);
	attsSymantec.push_back(uri_tld);
	attrCnt++;
	/* "x_p0f_detail" : "XP/2000" */
	RecordAttribute *x_p0f_detail = new RecordAttribute(attrCnt, path,
			"x_p0f_detail", stringType);
	attsSymantec.push_back(x_p0f_detail);
	attrCnt++;
	/* "x_p0f_genre" : "Windows" */
	RecordAttribute *x_p0f_genre = new RecordAttribute(attrCnt, path, "x_p0f_genre",
			stringType);
	attsSymantec.push_back(x_p0f_genre);
	attrCnt++;

	/* "x_p0f_signature" : "64380:116:1:48:M1460,N,N,S:." */
	RecordAttribute *x_p0f_signature = new RecordAttribute(attrCnt, path,
			"x_p0f_signature", stringType);
	attsSymantec.push_back(x_p0f_signature);
	attrCnt++;

	RecordType symantecRec = RecordType(attsSymantec);
	symantec.recType = symantecRec;

	datasetCatalog["symantec"] = symantec;
}

/* TRUNK */
void symantecCoreIDDatesSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset symantec;

	#ifdef SYMANTEC_LOCAL
	string path = string("inputs/json/spam/spamsCoreIDDates100.json");
	symantec.linehint = 100;
	#endif
	#ifdef SYMANTEC_SERVER
//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreIDDates100.json");
//	symantec.linehint = 100;
	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsCoreIDDates28m-edit.json");
	symantec.linehint = 27991116;

	#endif
	symantec.path = path;

	list<RecordAttribute*> attsSymantec = list<RecordAttribute*>();

	int attrCnt = 1;
	RecordAttribute *id = new RecordAttribute(attrCnt, path, "id", intType);
	attsSymantec.push_back(id);
	attrCnt++;
	/* "IP" : "83.149.45.128" */
	RecordAttribute *ip = new RecordAttribute(attrCnt, path, "IP", stringType);
	attsSymantec.push_back(ip);
	attrCnt++;
	/* "_id" : { "$oid" : "4ebbd37d466e8b0b55000000" } */
	RecordAttribute *oid = new RecordAttribute(1, path, "$oid", stringType);
	list<RecordAttribute*> oidNested = list<RecordAttribute*>();
	oidNested.push_back(oid);
	RecordType *idRec = new RecordType(oidNested);
	RecordAttribute *_id = new RecordAttribute(attrCnt, path, "_id", idRec);
	attsSymantec.push_back(_id);
	attrCnt++;
	/* "attach" : [ "whigmaleerie.jpg" ] (but tends to be empty) */
	ListType *attachList = new ListType(*stringType);
	RecordAttribute *attach = new RecordAttribute(attrCnt, path, "attach",
			attachList);
	attsSymantec.push_back(attach);
	attrCnt++;
	/* "body_txt_a" : "blablabla" */
	RecordAttribute *body_txt_a = new RecordAttribute(attrCnt, path, "body_txt_a",
			stringType);
	attsSymantec.push_back(body_txt_a);
	attrCnt++;
	/* "charset" : "windows-1252" */
	RecordAttribute *charset = new RecordAttribute(attrCnt, path, "charset",
			stringType);
	attsSymantec.push_back(charset);
	attrCnt++;
	/* "city" : "Ryazan" */
	RecordAttribute *city = new RecordAttribute(attrCnt, path, "city", stringType);
	attsSymantec.push_back(city);
	attrCnt++;
	/* "content_type" : [ "text/html", "text/plain" ] */
	ListType *contentList = new ListType(*stringType);
	RecordAttribute *content_type = new RecordAttribute(attrCnt, path,
			"content_type", contentList);
	attsSymantec.push_back(content_type);
	attrCnt++;
	/* "country_code" : "RU" */
	RecordAttribute *country_code = new RecordAttribute(attrCnt, path,
			"country_code", stringType);
	attsSymantec.push_back(country_code);
	attrCnt++;
	/* "cte" : "unknown" */
	RecordAttribute *cte = new RecordAttribute(attrCnt, path, "cte", stringType);
	attsSymantec.push_back(cte);


	/* "date" : { "$date" : 1285919417000 } */
	RecordAttribute *date_ = new RecordAttribute(1, path, "$date", floatType);
	list<RecordAttribute*> dateNested = list<RecordAttribute*>();
	dateNested.push_back(date_);
	RecordType *dateRec = new RecordType(dateNested);
	RecordAttribute *date = new RecordAttribute(attrCnt, path, "date", dateRec);
	attsSymantec.push_back(date);
	attrCnt++;

	/*"day":{"year":2010,"month":10,"day":1}*/
	RecordAttribute *year_ = new RecordAttribute(1, path, "year", intType);
	RecordAttribute *month_ = new RecordAttribute(2, path, "month", intType);
	RecordAttribute *day_ = new RecordAttribute(3, path, "day", intType);
	list<RecordAttribute*> dayNested = list<RecordAttribute*>();
	dayNested.push_back(year_);
	dayNested.push_back(month_);
	dayNested.push_back(day_);
	RecordType *dayRec = new RecordType(dayNested);
	RecordAttribute *day = new RecordAttribute(attrCnt, path, "day", dayRec);
	attsSymantec.push_back(day);
	attrCnt++;

	/* "from_domain" : "domain733674.com" */
	RecordAttribute *from_domain = new RecordAttribute(attrCnt, path, "from_domain",
			stringType);
	attsSymantec.push_back(from_domain);
	attrCnt++;
	/* "host" : "airtelbroadband.in (but tends to be empty) */
	RecordAttribute *host = new RecordAttribute(attrCnt, path, "host", stringType);
	attsSymantec.push_back(host);
	attrCnt++;
	/* "lang" : "english" */
	RecordAttribute *lang = new RecordAttribute(attrCnt, path, "lang", stringType);
	attsSymantec.push_back(lang);
	attrCnt++;
	/* "lat" : 54.6197 */
	RecordAttribute *lat = new RecordAttribute(attrCnt, path, "lat", floatType);
	attsSymantec.push_back(lat);
	attrCnt++;
	/* "long" : 39.74 */
	RecordAttribute *long_ = new RecordAttribute(attrCnt, path, "long", floatType);
	attsSymantec.push_back(long_);
	attrCnt++;
	/* "rcpt_domain" : "domain555065.com" */
	RecordAttribute *rcpt_domain = new RecordAttribute(attrCnt, path, "rcpt_domain",
			stringType);
	attsSymantec.push_back(rcpt_domain);
	attrCnt++;
	/* "size" : 3712 */
	RecordAttribute *size = new RecordAttribute(attrCnt, path, "size", intType);
	attsSymantec.push_back(size);
	attrCnt++;
	/* "subject" : "LinkedIn Messages, 9/30/2010" */
	RecordAttribute *subject = new RecordAttribute(attrCnt, path, "subject",
			stringType);
	attsSymantec.push_back(subject);
	attrCnt++;
	/* "uri" : [ "http://hetfonteintje.com/1.html" ] */
	ListType *uriList = new ListType(*stringType);
	RecordAttribute *uri = new RecordAttribute(attrCnt, path, "uri", uriList);
	attsSymantec.push_back(uri);
	attrCnt++;
	/* "uri_domain" : [ "hetfonteintje.com" ] */
	ListType *domainList = new ListType(*stringType);
	RecordAttribute *domain = new RecordAttribute(attrCnt, path, "uri_domain",
			domainList);
	attsSymantec.push_back(domain);
	/* "uri_tld" : [ ".com" ] */
	ListType *tldList = new ListType(*stringType);
	RecordAttribute *uri_tld = new RecordAttribute(attrCnt, path, "uri_tld",
			tldList);
	attsSymantec.push_back(uri_tld);
	attrCnt++;
	/* "x_p0f_detail" : "XP/2000" */
	RecordAttribute *x_p0f_detail = new RecordAttribute(attrCnt, path,
			"x_p0f_detail", stringType);
	attsSymantec.push_back(x_p0f_detail);
	attrCnt++;
	/* "x_p0f_genre" : "Windows" */
	RecordAttribute *x_p0f_genre = new RecordAttribute(attrCnt, path, "x_p0f_genre",
			stringType);
	attsSymantec.push_back(x_p0f_genre);
	attrCnt++;

	/* "x_p0f_signature" : "64380:116:1:48:M1460,N,N,S:." */
	RecordAttribute *x_p0f_signature = new RecordAttribute(attrCnt, path,
			"x_p0f_signature", stringType);
	attsSymantec.push_back(x_p0f_signature);
	attrCnt++;

	RecordType symantecRec = RecordType(attsSymantec);
	symantec.recType = symantecRec;

	datasetCatalog["symantecIDDates"] = symantec;
}

void symantecBinSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset symantecBin;

	#ifdef SYMANTEC_LOCAL
	string path = string("inputs/json/spam/col/symantec");
	symantecBin.linehint = 100;
	#endif
	#ifdef SYMANTEC_SERVER
	string path = string("/cloud_store/manosk/data/vida-engine/symantec/col/symantec");
	symantecBin.linehint = 500000000;
	#endif
	symantecBin.path = path;

	list<RecordAttribute*> attsSymantec = list<RecordAttribute*>();

	int attrCnt = 1;
	RecordAttribute *id = new RecordAttribute(attrCnt, path, "id", intType);
	attsSymantec.push_back(id);
	attrCnt++;
	/* dim */
	RecordAttribute *dim = new RecordAttribute(attrCnt, path, "dim", intType);
	attsSymantec.push_back(dim);
	attrCnt++;
	/* dataset */
	RecordAttribute *dataset = new RecordAttribute(attrCnt, path, "dataset", stringType);
	attsSymantec.push_back(dataset);
	attrCnt++;
	/* analysis */
	RecordAttribute *analysis = new RecordAttribute(attrCnt, path, "analysis", stringType);
	attsSymantec.push_back(analysis);
	attrCnt++;
	/* slice_id */
	RecordAttribute *slice_id = new RecordAttribute(attrCnt, path, "slice_id", intType);
	attsSymantec.push_back(slice_id);
	attrCnt++;
	/* cluster */
	RecordAttribute *cluster = new RecordAttribute(attrCnt, path, "cluster", intType);
	attsSymantec.push_back(cluster);
	attrCnt++;
	/* cluster */
	RecordAttribute *mdc = new RecordAttribute(attrCnt, path, "mdc", intType);
	attsSymantec.push_back(mdc);
	attrCnt++;
	/* neighbors */
	RecordAttribute *neighbors = new RecordAttribute(attrCnt, path, "neighbors", intType);
	attsSymantec.push_back(neighbors);
	attrCnt++;
	/* size */
	RecordAttribute *size = new RecordAttribute(attrCnt, path, "size", intType);
	attsSymantec.push_back(size);
	attrCnt++;
	/* cp */
	RecordAttribute *cp = new RecordAttribute(attrCnt, path, "cp", stringType);
	attsSymantec.push_back(cp);
	attrCnt++;
	/* p_event */
	RecordAttribute *p_event = new RecordAttribute(attrCnt, path, "p_event", floatType);
	attsSymantec.push_back(p_event);
	attrCnt++;
	/* value */
	RecordAttribute *value = new RecordAttribute(attrCnt, path, "value", floatType);
	attsSymantec.push_back(value);
	attrCnt++;
	/* pattern */
	RecordAttribute *pattern = new RecordAttribute(attrCnt, path, "pattern", stringType);
	attsSymantec.push_back(pattern);
	attrCnt++;

	RecordType symantecBinRec = RecordType(attsSymantec);
	symantecBin.recType = symantecBinRec;

	datasetCatalog["symantecBin"] = symantecBin;
}

/* Careful: Strings have no brackets! */
void symantecCSVSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	dataset symantecCSV;

	#ifdef SYMANTEC_LOCAL
	string path = string("inputs/json/spam/spamsClasses1000-unordered-nocomma.csv");
	symantecCSV.linehint = 1000;
	#endif
	#ifdef SYMANTEC_SERVER
//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsClasses1000-unordered-nocomma.csv");
//	symantecCSV.linehint = 1000;
	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsClasses400m-unordered-nocomma.csv");
	symantecCSV.linehint = 400000000;
//	string path = string("/cloud_store/manosk/data/vida-engine/symantec/spamsClasses1000-unordered.csv");
//	symantecCSV.linehint = 1000;
	#endif
	symantecCSV.path = path;

	list<RecordAttribute*> attsSymantec = list<RecordAttribute*>();

	int attrCnt = 1;
	RecordAttribute *id = new RecordAttribute(attrCnt, path, "id", intType);
	attsSymantec.push_back(id);
	attrCnt++;
	/* classa */
	RecordAttribute *classa = new RecordAttribute(attrCnt, path, "classa", intType);
	attsSymantec.push_back(classa);
	attrCnt++;
	/* classb */
	RecordAttribute *classb = new RecordAttribute(attrCnt, path, "classb", floatType);
	attsSymantec.push_back(classb);
	attrCnt++;
	/* city */
	RecordAttribute *city = new RecordAttribute(attrCnt, path, "city", stringType);
	attsSymantec.push_back(city);
	attrCnt++;
	/* country */
	RecordAttribute *country = new RecordAttribute(attrCnt, path, "country", stringType);
	attsSymantec.push_back(country);
	attrCnt++;
	/* country_code */
	RecordAttribute *country_code = new RecordAttribute(attrCnt, path, "country_code", stringType);
	attsSymantec.push_back(country_code);
	attrCnt++;
	/* size */
	RecordAttribute *size = new RecordAttribute(attrCnt, path, "size", intType);
	attsSymantec.push_back(size);
	attrCnt++;
	/* bot */
	RecordAttribute *bot = new RecordAttribute(attrCnt, path, "bot", stringType);
	attsSymantec.push_back(bot);
	attrCnt++;

	RecordType symantecBinRec = RecordType(attsSymantec);
	symantecCSV.recType = symantecBinRec;

	datasetCatalog["symantecCSV"] = symantecCSV;
}

#endif /* SYMANTEC_CONFIG_HPP_ */
