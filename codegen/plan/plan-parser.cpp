#include "plan/plan-parser.hpp"
#include "plugins/gpu-col-scan-plugin.hpp"
#include "plugins/scan-to-blocks-sm-plugin.hpp"
#ifndef NCUDA
#include "operators/gpu/gpu-join.hpp"
#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "operators/gpu/gpu-hash-group-by-chained.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/cpu-to-gpu.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#endif
#include "operators/hash-join-chained.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/mem-broadcast-device.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/mem-move-local-to.hpp"
#include "operators/exchange.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/block-to-tuples.hpp"
#include "operators/flush.hpp"
#include "operators/project.hpp"
#include "operators/sort.hpp"
#include "operators/gpu/gpu-sort.hpp"
#include "operators/unionall.hpp"
#include "operators/split.hpp"
#include "operators/dict-scan.hpp"

#include "rapidjson/error/en.h"
/* too primitive */
struct PlanHandler {
    bool Null() { cout << "Null()" << endl; return true; }
    bool Bool(bool b) { cout << "Bool(" << std::boolalpha << b << ")" << endl; return true; }
    bool Int(int i) { cout << "Int(" << i << ")" << endl; return true; }
    bool Uint(unsigned u) { cout << "Uint(" << u << ")" << endl; return true; }
    bool Int64(int64_t i) { cout << "Int64(" << i << ")" << endl; return true; }
    bool Uint64(uint64_t u) { cout << "Uint64(" << u << ")" << endl; return true; }
    bool Double(double d) { cout << "Double(" << d << ")" << endl; return true; }
    bool String(const char* str, SizeType length, bool copy) {
        cout << "String(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
        return true;
    }
    bool StartObject() { cout << "StartObject()" << endl; return true; }
    bool Key(const char* str, SizeType length, bool copy) {
        cout << "Key(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
        return true;
    }
    bool EndObject(SizeType memberCount) { cout << "EndObject(" << memberCount << ")" << endl; return true; }
    bool StartArray() { cout << "StartArray()" << endl; return true; }
    bool EndArray(SizeType elementCount) { cout << "EndArray(" << elementCount << ")" << endl; return true; }
};

PlanExecutor::PlanExecutor(const char *planPath, CatalogParser& cat, const char *moduleName) :
		planPath(planPath), moduleName(moduleName), catalogParser(cat), exprParser(cat) {

	/* Init LLVM Context and catalog */
	ctx = prepareContext(this->moduleName);
	RawCatalog& catalog = RawCatalog::getInstance();

	//Input Path
	const char* nameJSON = planPath;
	//Prepare Input
	struct stat statbuf;
	stat(nameJSON, &statbuf);
	size_t fsize = statbuf.st_size;

	int fd = open(nameJSON, O_RDONLY);
	if (fd == -1) {
		throw runtime_error(string("json.open"));
	}

	const char *bufJSON = (const char*) mmap(NULL, fsize, PROT_READ,
			MAP_PRIVATE, fd, 0);
	if (bufJSON == MAP_FAILED ) {
		const char *err = "json.mmap";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	Document document; // Default template parameter uses UTF8 and MemoryPoolAllocator.
	if (document.Parse(bufJSON).HasParseError()) {
		const char *err = "[PlanExecutor: ] Error parsing physical plan";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	/* Start plan traversal. */
	printf("\nParsing physical plan:\n");
	assert(document.IsObject());

	assert(document.HasMember("operator"));
	assert(document["operator"].IsString());
	printf("operator = %s\n", document["operator"].GetString());

	parsePlan(document);

	vector<Plugin*>::iterator pgIter = activePlugins.begin();

	/* Cleanup */
	for(; pgIter != activePlugins.end(); pgIter++)	{
		Plugin *currPg = *pgIter;
		currPg->finish();
	}

	return;
}

PlanExecutor::PlanExecutor(const char *planPath, CatalogParser& cat, const char *moduleName, RawContext * ctx) :
		planPath(planPath), moduleName(moduleName), catalogParser(cat), ctx(ctx), exprParser(cat) {
	RawCatalog& catalog = RawCatalog::getInstance();

	//Input Path
	const char* nameJSON = planPath;
	//Prepare Input
	struct stat statbuf;
	stat(nameJSON, &statbuf);
	size_t fsize = statbuf.st_size;

	int fd = open(nameJSON, O_RDONLY);
	if (fd == -1) {
		throw runtime_error(string("json.open"));
	}

	const char *bufJSON = (const char*) mmap(NULL, fsize, PROT_READ,
			MAP_PRIVATE, fd, 0);
	if (bufJSON == MAP_FAILED ) {
		const char *err = "json.mmap";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	Document document; // Default template parameter uses UTF8 and MemoryPoolAllocator.
	auto & parsed = document.Parse(bufJSON);
	if (parsed.HasParseError()) {
		ParseResult ok = (ParseResult) parsed;
		fprintf(stderr, "JSON parse error: %s (%lu)", RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset());
		const char *err = "[PlanExecutor: ] Error parsing physical plan (JSON parsing error)";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	/* Start plan traversal. */
	printf("\nParsing physical plan:\n");
	assert(document.IsObject());

	assert(document.HasMember("operator"));
	assert(document["operator"].IsString());
	printf("operator = %s\n", document["operator"].GetString());

	parsePlan(document, false);

	// vector<Plugin*>::iterator pgIter = activePlugins.begin();

	// /* Cleanup */
	// for(; pgIter != activePlugins.end(); pgIter++)	{
	// 	Plugin *currPg = *pgIter;
	// 	currPg->finish();
	// }

	return;
}


void PlanExecutor::parsePlan(const rapidjson::Document& doc, bool execute)	{
	splitOps.clear();
	RawOperator* planRootOp = parseOperator(doc);

	planRootOp->produce();

	//Run function
	ctx->prepareFunction(ctx->getGlobalFunction());

	if (execute){
		RawCatalog& catalog = RawCatalog::getInstance();
		/* XXX Remove when testing caches (?) */
		catalog.clear();
	}
}

void PlanExecutor::cleanUp(){
	RawCatalog& catalog = RawCatalog::getInstance();
	/* XXX Remove when testing caches (?) */
	catalog.clear();

	vector<Plugin*>::iterator pgIter = activePlugins.begin();

	/* Cleanup */
	for(; pgIter != activePlugins.end(); pgIter++)	{
		Plugin *currPg = *pgIter;
		currPg->finish();
	}
}

RawOperator* PlanExecutor::parseOperator(const rapidjson::Value& val)	{

	const char *keyPg = "plugin";
	const char *keyOp = "operator";

	assert(val.HasMember(keyOp));
	assert(val[keyOp].IsString());
	const char *opName = val["operator"].GetString();

	RawOperator *newOp = NULL;

	if (strcmp(opName, "reduce") == 0) {
		/* "Multi - reduce"! */
		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/* get monoid(s) */
		assert(val.HasMember("accumulator"));
		assert(val["accumulator"].IsArray());
		vector<Monoid> accs;
		const rapidjson::Value& accsJSON = val["accumulator"];
		for (SizeType i = 0; i < accsJSON.Size(); i++) // rapidjson uses SizeType instead of size_t.
		{
			assert(accsJSON[i].IsString());
			Monoid acc = parseAccumulator(accsJSON[i].GetString());
			accs.push_back(acc);
		}

		/*
		 * parse output expressions
		 * XXX Careful: Assuming numerous output expressions!
		 */
		assert(val.HasMember("e"));
		assert(val["e"].IsArray());
		vector<expressions::Expression*> e;
		const rapidjson::Value& exprsJSON = val["e"];
		for (SizeType i = 0; i < exprsJSON.Size(); i++)
		{
			expressions::Expression *outExpr = parseExpression(exprsJSON[i]);
			e.push_back(outExpr);
		}

		/* parse filtering expression */
		assert(val.HasMember("p"));
		assert(val["p"].IsObject());
		expressions::Expression *p = parseExpression(val["p"]);

		/* 'Multi-reduce' used */
#ifndef NCUDA
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new opt::GpuReduce(accs, e, p, childOp, dynamic_cast<GpuRawContext *>(this->ctx));
		} else {
#endif
			newOp = new opt::Reduce(accs, e, p, childOp, this->ctx,true,moduleName);
#ifndef NCUDA
		}
#endif
		childOp->setParent(newOp);
	} else if (strcmp(opName, "print") == 0) {
		/* "Multi - reduce"! */
		if (val.HasMember("plugin")){
			assert(val["plugin"].IsObject());
			parsePlugin(val["plugin"]);
		}

		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/*
		 * parse output expressions
		 * XXX Careful: Assuming numerous output expressions!
		 */
		assert(val.HasMember("e"));
		assert(val["e"].IsArray());
		vector<expressions::Expression*> e;
		const rapidjson::Value& exprsJSON = val["e"];
		for (SizeType i = 0; i < exprsJSON.Size(); i++)
		{
			expressions::Expression *outExpr = parseExpression(exprsJSON[i]);
			e.push_back(outExpr);
		}

		newOp = new Flush(e, childOp, this->ctx, moduleName);
		childOp->setParent(newOp);
	} else if (strcmp(opName, "sort") == 0) {
		/* "Multi - reduce"! */
		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/*
		 * parse output expressions
		 * XXX Careful: Assuming numerous output expressions!
		 */
		assert(val.HasMember("e"));
		assert(val["e"].IsArray());
		vector<expressions::Expression *> e;
		vector<direction                > d;
		vector<RecordAttribute         *> recattr;
		const rapidjson::Value& exprsJSON = val["e"];
		for (SizeType i = 0; i < exprsJSON.Size(); i++)
		{	
			assert(exprsJSON[i].IsObject());
			assert(exprsJSON[i].HasMember("expression"));
			assert(exprsJSON[i]["expression"].IsObject());
			expressions::Expression *outExpr = parseExpression(exprsJSON[i]["expression"]);
			e.push_back(outExpr);
			assert(exprsJSON[i].HasMember("direction"));
			assert(exprsJSON[i]["direction"].IsString());
			std::string dir = exprsJSON[i]["direction"].GetString();
			if      (dir == "ASC" ) d.push_back(ASC );
			else if (dir == "NONE") d.push_back(NONE);
			else if (dir == "DESC") d.push_back(DESC);
			else 					assert(false);


			recattr.push_back(new RecordAttribute{outExpr->getRegisteredAs()});
		}

		std::string relName = e[0]->getRegisteredRelName();

		InputInfo * datasetInfo = (this->catalogParser).getOrCreateInputInfo(relName);
		RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};
		RecordAttribute * reg_as = new RecordAttribute(relName, "__sorted", new RecordType(recattr)); 
		std::cout << "Registered: " << reg_as->getRelationName() << "." << reg_as->getAttrName() << std::endl;
		rec->appendAttribute(reg_as);

		datasetInfo->exprType = new BagType{*rec};

#ifndef NCUDA
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			gran_t granularity = gran_t::GRID;

			if (val.HasMember("granularity")){
				assert(val["granularity"].IsString());
				std::string gran = val["granularity"].GetString();
				std::transform(gran.begin(), gran.end(), gran.begin(), [](unsigned char c){ return std::tolower(c); });
				if      (gran == "grid"  ) granularity = gran_t::GRID;
				else if (gran == "block" ) granularity = gran_t::BLOCK;
				else if (gran == "thread") granularity = gran_t::THREAD;
				else 	assert(false && "granularity must be one of GRID, BLOCK, THREAD");
			}

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new GpuSort(childOp, dynamic_cast<GpuRawContext *>(this->ctx), e, d, granularity);
		} else {
#endif
			newOp = new Sort(childOp, dynamic_cast<GpuRawContext *>(this->ctx), e, d);
#ifndef NCUDA
		}
#endif
		childOp->setParent(newOp);
	} else if (strcmp(opName, "project") == 0) {
		/* "Multi - reduce"! */
		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/*
		 * parse output expressions
		 * XXX Careful: Assuming numerous output expressions!
		 */
		assert(val.HasMember("e"));
		assert(val["e"].IsArray());
		vector<expressions::Expression*> e;
		const rapidjson::Value& exprsJSON = val["e"];
		for (SizeType i = 0; i < exprsJSON.Size(); i++){
			expressions::Expression *outExpr = parseExpression(exprsJSON[i]);
			e.push_back(outExpr);
		}

		assert(val.HasMember("relName"));
		assert(val["relName"].IsString());

		newOp = new Project(e, val["relName"].GetString(), childOp, this->ctx);
		childOp->setParent(newOp);
	} else if (strcmp(opName, "unnest") == 0) {

		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/* parse filtering expression */
		assert(val.HasMember("p"));
		assert(val["p"].IsObject());
		expressions::Expression *p = parseExpression(val["p"]);

		/* parse path expression */
		assert(val.HasMember("path"));
		assert(val["path"].IsObject());


		assert(val["path"]["e"].IsObject());
		expressions::Expression *exprToUnnest = parseExpression(
				val["path"]["e"]);
		expressions::RecordProjection *proj =
				dynamic_cast<expressions::RecordProjection*>(exprToUnnest);
		if (proj == NULL) {
			string error_msg = string(
					"[Unnest: ] Cannot cast to record projection");
			LOG(ERROR)<< error_msg;
			throw runtime_error(string(error_msg));
		}
		
		string pathAlias;
		if (exprToUnnest->isRegistered()){
			pathAlias = exprToUnnest->getRegisteredAttrName();
		} else {
			assert(val["path"]["name"].IsString());
			pathAlias = val["path"]["name"].GetString();
		}

		Path *projPath = new Path(pathAlias,proj);

		newOp = new Unnest(p, *projPath, childOp);
		childOp->setParent(newOp);
	} else if (strcmp(opName, "outer_unnest") == 0) {

		/* parse operator input */
		RawOperator* childOp = parseOperator(val["input"]);

		/* parse filtering expression */
		assert(val.HasMember("p"));
		assert(val["p"].IsObject());
		expressions::Expression *p = parseExpression(val["p"]);

		/* parse path expression */
		assert(val.HasMember("path"));
		assert(val["path"].IsObject());

		assert(val["path"]["name"].IsString());
		string pathAlias = val["path"]["name"].GetString();

		assert(val["path"]["e"].IsObject());
		expressions::Expression *exprToUnnest = parseExpression(
				val["path"]["e"]);
		expressions::RecordProjection *proj =
				dynamic_cast<expressions::RecordProjection*>(exprToUnnest);
		if (proj == NULL) {
			string error_msg = string(
					"[Unnest: ] Cannot cast to record projection");
			LOG(ERROR)<< error_msg;
			throw runtime_error(string(error_msg));
		}

		Path *projPath = new Path(pathAlias, proj);

		newOp = new OuterUnnest(p, *projPath, childOp);
		childOp->setParent(newOp);
	} else if(strcmp(opName, "groupby") == 0 || strcmp(opName, "hashgroupby-chained") == 0)	{
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* child = parseOperator(val["input"]);

#ifndef NCUDA
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			assert(val.HasMember("hash_bits"));
			assert(val["hash_bits"].IsInt());
			int hash_bits = val["hash_bits"].GetInt();

			// assert(val.HasMember("w"));
			// assert(val["w"].IsArray());
			// vector<size_t> widths;

			// const rapidjson::Value& wJSON = val["w"];
			// for (SizeType i = 0; i < wJSON.Size(); i++){
			// 	assert(wJSON[i].IsInt());
			// 	widths.push_back(wJSON[i].GetInt());
			// }

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("e"));
			assert(val["e"].IsArray());
			vector<GpuAggrMatExpr> e;
			const rapidjson::Value& aggrJSON = val["e"];
			for (SizeType i = 0; i < aggrJSON.Size(); i++){
				assert(aggrJSON[i].HasMember("e"     ));
				assert(aggrJSON[i].HasMember("m"     ));
				assert(aggrJSON[i]["m"].IsString()    );
				assert(aggrJSON[i].HasMember("packet"));
				assert(aggrJSON[i]["packet"].IsInt()  );
				assert(aggrJSON[i].HasMember("offset"));
				assert(aggrJSON[i]["offset"].IsInt()  );
				expressions::Expression *outExpr = parseExpression(aggrJSON[i]["e"]);

				e.emplace_back(outExpr, aggrJSON[i]["packet"].GetInt(), aggrJSON[i]["offset"].GetInt(), parseAccumulator(aggrJSON[i]["m"].GetString()));
			}

			assert(val.HasMember("k"));
			assert(val["k"].IsArray());
			vector<expressions::Expression *> key_expr;
			const rapidjson::Value& keyJSON = val["k"];
			for (SizeType i = 0; i < keyJSON.Size(); i++){
				key_expr.emplace_back(parseExpression(keyJSON[i]));
			}

			assert(val.HasMember("maxInputSize"));
			assert(val["maxInputSize"].IsUint64());
			
			size_t maxInputSize = val["maxInputSize"].GetUint64();

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			// newOp = new GpuHashGroupByChained(e, widths, key_expr, child, hash_bits,
			// 					dynamic_cast<GpuRawContext *>(this->ctx), maxInputSize);
			newOp = new GpuHashGroupByChained(e, key_expr, child, hash_bits,
								dynamic_cast<GpuRawContext *>(this->ctx), maxInputSize);
		} else {
#endif
			assert(val.HasMember("hash_bits"));
			assert(val["hash_bits"].IsInt());
			int hash_bits = val["hash_bits"].GetInt();

			// assert(val.HasMember("w"));
			// assert(val["w"].IsArray());
			// vector<size_t> widths;

			// const rapidjson::Value& wJSON = val["w"];
			// for (SizeType i = 0; i < wJSON.Size(); i++){
			// 	assert(wJSON[i].IsInt());
			// 	widths.push_back(wJSON[i].GetInt());
			// }

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("e"));
			assert(val["e"].IsArray());
			// vector<GpuAggrMatExpr> e;
			const rapidjson::Value& aggrJSON = val["e"];
			vector<Monoid> accs;
			vector<string> aggrLabels;
			vector<expressions::Expression*> outputExprs;
			vector<expressions::Expression*> exprsToMat;
			vector<materialization_mode> outputModes;
			map<string, RecordAttribute*> mapOids;
			vector<RecordAttribute*> fieldsToMat;
			for (SizeType i = 0; i < aggrJSON.Size(); i++){
				assert(aggrJSON[i].HasMember("e"     ));
				assert(aggrJSON[i].HasMember("m"     ));
				assert(aggrJSON[i]["m"].IsString()    );
				assert(aggrJSON[i].HasMember("packet"));
				assert(aggrJSON[i]["packet"].IsInt()  );
				assert(aggrJSON[i].HasMember("offset"));
				assert(aggrJSON[i]["offset"].IsInt()  );
				expressions::Expression *outExpr = parseExpression(aggrJSON[i]["e"]);

				// e.emplace_back(outExpr, aggrJSON[i]["packet"].GetInt(), aggrJSON[i]["offset"].GetInt(), );

				aggrLabels.push_back(outExpr->getRegisteredAttrName());
				accs.push_back(parseAccumulator(aggrJSON[i]["m"].GetString()));

				outputExprs.push_back(outExpr);

				//XXX STRONG ASSUMPTION: Expression is actually a record projection!
				expressions::RecordProjection *proj =
						dynamic_cast<expressions::RecordProjection *>(outExpr);


				if (proj == NULL) {
					if (outExpr->getTypeID() != expressions::CONSTANT){
						string error_msg = string(
								"[Nest: ] Cannot cast to rec projection. Original: ")
								+ outExpr->getExpressionType()->getType();
						LOG(ERROR)<< error_msg;
						throw runtime_error(string(error_msg));
					}
				} else {
					exprsToMat.push_back(outExpr);
					outputModes.insert(outputModes.begin(), EAGER);

					//Added in 'wanted fields'
					RecordAttribute *recAttr = new RecordAttribute(proj->getAttribute());
					fieldsToMat.push_back(new RecordAttribute(outExpr->getRegisteredAs()));

					string relName = recAttr->getRelationName();
					if (mapOids.find(relName) == mapOids.end()) {
						InputInfo *datasetInfo =
								(this->catalogParser).getInputInfo(relName);
						RecordAttribute *oid =
								new RecordAttribute(recAttr->getRelationName(), activeLoop,
								datasetInfo->oidType);
						mapOids[relName] = oid;
						expressions::RecordProjection *oidProj =
								new expressions::RecordProjection(outExpr, *oid);
						//Added in 'wanted expressions'
						LOG(INFO)<< "[Plan Parser: ] Injecting OID for " << relName;
						std::cout << "[Plan Parser: ] Injecting OID for " << relName << std::endl;
						/* ORDER OF expression fields matters!! OIDs need to be placed first! */
						exprsToMat.insert(exprsToMat.begin(), oidProj);
						outputModes.insert(outputModes.begin(), EAGER);
					}
				}
			}

			/* Predicate */
			expressions::Expression * predExpr = new expressions::BoolConstant(true);

			assert(val.HasMember("k"));
			assert(val["k"].IsArray());
			vector<expressions::Expression *> key_expr;
			const rapidjson::Value& keyJSON = val["k"];
			for (SizeType i = 0; i < keyJSON.Size(); i++){
				key_expr.emplace_back(parseExpression(keyJSON[i]));
			}

			assert(val.HasMember("maxInputSize"));
			assert(val["maxInputSize"].IsUint64());
			
			size_t maxInputSize = val["maxInputSize"].GetUint64();



			const char *keyGroup = "f";
			const char *keyNull  = "g";
			const char *keyPred  = "p";
			const char *keyExprs = "e";
			const char *keyAccum = "accumulator";
			/* Physical Level Info */
			const char *keyAggrNames = "aggrLabels";
			//Materializer
			const char *keyMat = "fields";

			/* Group By */
			// assert(key_expr.size() == 1);
			// expressions::Expression *groupByExpr = key_expr[0];

			for (const auto &e: key_expr){
				//XXX STRONG ASSUMPTION: Expression is actually a record projection!
				expressions::RecordProjection *proj =
						dynamic_cast<expressions::RecordProjection *>(e);

				if (proj == NULL) {
					if (e->getTypeID() != expressions::CONSTANT){
						string error_msg = string(
								"[Nest: ] Cannot cast to rec projection. Original: ")
								+ e->getExpressionType()->getType();
						LOG(ERROR)<< error_msg;
						throw runtime_error(string(error_msg));
					}
				} else {
					exprsToMat.push_back(e);
					outputModes.insert(outputModes.begin(), EAGER);

					//Added in 'wanted fields'
					RecordAttribute *recAttr = new RecordAttribute(proj->getAttribute());
					fieldsToMat.push_back(new RecordAttribute(e->getRegisteredAs()));

					string relName = recAttr->getRelationName();
					if (mapOids.find(relName) == mapOids.end()) {
						InputInfo *datasetInfo =
								(this->catalogParser).getInputInfo(relName);
						RecordAttribute *oid =
								new RecordAttribute(recAttr->getRelationName(), activeLoop,
								datasetInfo->oidType);
						std::cout << datasetInfo->oidType->getType() << std::endl;
						mapOids[relName] = oid;
						expressions::RecordProjection *oidProj =
								new expressions::RecordProjection(e, *oid);
						//Added in 'wanted expressions'
						LOG(INFO)<< "[Plan Parser: ] Injecting OID for " << relName;
						std::cout << "[Plan Parser: ] Injecting OID for " << relName << std::endl;
						/* ORDER OF expression fields matters!! OIDs need to be placed first! */
						exprsToMat.insert(exprsToMat.begin(), oidProj);
						outputModes.insert(outputModes.begin(), EAGER);
					}
				}
			}
			/* Null-to-zero Checks */
			//FIXME not used in radix nest yet!
			// assert(val.HasMember(keyNull));
			// assert(val[keyNull].IsObject());
			expressions::Expression *nullsToZerosExpr = NULL;//parseExpression(val[keyNull]);

			/* Output aggregate expression(s) */
			// assert(val.HasMember(keyExprs));
			// assert(val[keyExprs].IsArray());
			// vector<expressions::Expression*> outputExprs;
			// for (SizeType i = 0; i < val[keyExprs].Size(); i++) {
			// 	expressions::Expression *expr = parseExpression(val[keyExprs][i]);
			// }

			/*
			 * *** WHAT TO MATERIALIZE ***
			 * XXX v0: JSON file contains a list of **RecordProjections**
			 * EXPLICIT OIDs injected by PARSER (not in json file by default)
			 * XXX Eager materialization atm
			 *
			 * XXX Why am I not using minimal constructor for materializer yet, as use cases do?
			 * 	-> Because then I would have to encode the OID type in JSON -> can be messy
			 */
			vector<RecordAttribute*> oids;
			MapToVec(mapOids, oids);
			/* FIXME This constructor breaks nest use cases that trigger caching */
			/* Check similar hook in radix-nest.cpp */
	//		Materializer *mat =
	//				new Materializer(fieldsToMat, exprsToMat, oids, outputModes);
			// for (const auto &e: exprsToMat) {
			// 	std::cout << "mat: " << e->getRegisteredRelName() << " " << e->getRegisteredAttrName() << std::endl;
			// }
			Materializer* matCoarse = new Materializer(fieldsToMat, exprsToMat,
					oids, outputModes);

			//Put operator together
			const char *opLabel = outputExprs[0]->getRegisteredRelName().c_str();
			std::cout << "regRelNAme" << opLabel << std::endl;
			newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels, predExpr,
					key_expr, nullsToZerosExpr, child, opLabel, *matCoarse);
#ifndef NCUDA
		}
#endif
		child->setParent(newOp);
	} else if(strcmp(opName, "hashjoin-chained") == 0)	{
		/* parse operator input */
		assert(val.HasMember("probe_input"));
		assert(val["probe_input"].IsObject());
		RawOperator* probe_op = parseOperator(val["probe_input"]);
		/* parse operator input */
		assert(val.HasMember("build_input"));
		assert(val["build_input"].IsObject());
		RawOperator* build_op = parseOperator(val["build_input"]);

		assert(val.HasMember("build_k"));
		expressions::Expression *build_key_expr = parseExpression(val["build_k"]);

		assert(val.HasMember("probe_k"));
		expressions::Expression *probe_key_expr = parseExpression(val["probe_k"]);

// #ifndef NCUDA
// 		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			assert(val.HasMember("hash_bits"));
			assert(val["hash_bits"].IsInt());
			int hash_bits = val["hash_bits"].GetInt();

			assert(val.HasMember("build_w"));
			assert(val["build_w"].IsArray());
			vector<size_t> build_widths;

			const rapidjson::Value& build_wJSON = val["build_w"];
			for (SizeType i = 0; i < build_wJSON.Size(); i++){
				assert(build_wJSON[i].IsInt());
				build_widths.push_back(build_wJSON[i].GetInt());
			}

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("build_e"));
			assert(val["build_e"].IsArray());
			vector<GpuMatExpr> build_e;
			const rapidjson::Value& build_exprsJSON = val["build_e"];
			for (SizeType i = 0; i < build_exprsJSON.Size(); i++){
				assert(build_exprsJSON[i].HasMember("e"     ));
				assert(build_exprsJSON[i].HasMember("packet"));
				assert(build_exprsJSON[i]["packet"].IsInt());
				assert(build_exprsJSON[i].HasMember("offset"));
				assert(build_exprsJSON[i]["offset"].IsInt());
				expressions::Expression *outExpr = parseExpression(build_exprsJSON[i]["e"]);

				build_e.emplace_back(outExpr, build_exprsJSON[i]["packet"].GetInt(), build_exprsJSON[i]["offset"].GetInt());
			}

			assert(val.HasMember("probe_w"));
			assert(val["probe_w"].IsArray());
			vector<size_t> probe_widths;

			const rapidjson::Value& probe_wJSON = val["probe_w"];
			for (SizeType i = 0; i < probe_wJSON.Size(); i++){
				assert(probe_wJSON[i].IsInt());
				probe_widths.push_back(probe_wJSON[i].GetInt());
			}

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("probe_e"));
			assert(val["probe_e"].IsArray());
			vector<GpuMatExpr> probe_e;
			const rapidjson::Value& probe_exprsJSON = val["probe_e"];
			for (SizeType i = 0; i < probe_exprsJSON.Size(); i++){
				assert(probe_exprsJSON[i].HasMember("e"     ));
				assert(probe_exprsJSON[i].HasMember("packet"));
				assert(probe_exprsJSON[i]["packet"].IsInt());
				assert(probe_exprsJSON[i].HasMember("offset"));
				assert(probe_exprsJSON[i]["offset"].IsInt());
				expressions::Expression *outExpr = parseExpression(probe_exprsJSON[i]["e"]);

				probe_e.emplace_back(outExpr, probe_exprsJSON[i]["packet"].GetInt(), probe_exprsJSON[i]["offset"].GetInt());
			}

			assert(val.HasMember("maxBuildInputSize"));
			assert(val["maxBuildInputSize"].IsUint64());
			
			size_t maxBuildInputSize = val["maxBuildInputSize"].GetUint64();

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
#ifndef NCUDA
			if (val.HasMember("gpu") && val["gpu"].GetBool()){
				newOp = new GpuHashJoinChained(build_e, build_widths, build_key_expr, build_op,
								probe_e, probe_widths, probe_key_expr, probe_op, hash_bits,
								dynamic_cast<GpuRawContext *>(this->ctx), maxBuildInputSize);
			} else {
#endif
				newOp = new HashJoinChained(build_e, build_widths, build_key_expr, build_op,
									probe_e, probe_widths, probe_key_expr, probe_op, hash_bits,
									dynamic_cast<GpuRawContext *>(this->ctx), maxBuildInputSize);
#ifndef NCUDA
			}
#endif
// 		} else {
// #endif
// 			expressions::BinaryExpression *predExpr = new expressions::EqExpression(build_key_expr, probe_key_expr);

// 			/*
// 			 * *** WHAT TO MATERIALIZE ***
// 			 * XXX v0: JSON file contains a list of **RecordProjections**
// 			 * EXPLICIT OIDs injected by PARSER (not in json file by default)
// 			 * XXX Eager materialization atm
// 			 *
// 			 * XXX Why am I not using minimal constructor for materializer yet, as use cases do?
// 			 * 	-> Because then I would have to encode the OID type in JSON -> can be messy
// 			 */

// 			//LEFT SIDE

// 			/*
// 			 * parse output expressions
// 			 * XXX Careful: Assuming numerous output expressions!
// 			 */
// 			assert(val.HasMember("build_e"));
// 			assert(val["build_e"].IsArray());
// 			vector<expressions::Expression *> exprBuild   ;
// 			vector<RecordAttribute         *> fieldsBuild ;
// 			map<string, RecordAttribute    *> mapOidsBuild;
// 			vector<materialization_mode>      outputModesBuild;

// 			{
// 				exprBuild.emplace_back(build_key_expr);
				
// 				expressions::Expression * exprR = exprBuild.back();

// 				outputModesBuild.insert(outputModesBuild.begin(), EAGER);

// 				expressions::RecordProjection *projBuild =
// 						dynamic_cast<expressions::RecordProjection *>(exprR);
// 				if(projBuild == NULL)
// 				{
// 					string error_msg = string(
// 							"[Join: ] Cannot cast to rec projection. Original: ")
// 							+ exprR->getExpressionType()->getType();
// 					LOG(ERROR)<< error_msg;
// 					throw runtime_error(string(error_msg));
// 				}

// 				//Added in 'wanted fields'
// 				RecordAttribute *recAttr = new RecordAttribute(projBuild->getAttribute());
// 				fieldsBuild.push_back(new RecordAttribute(exprR->getRegisteredAs()));

// 				string relName = recAttr->getRelationName();
// 				if (mapOidsBuild.find(relName) == mapOidsBuild.end()) {
// 					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
// 							relName);
// 					RecordAttribute *oid = new RecordAttribute(
// 							recAttr->getRelationName(), activeLoop,
// 							datasetInfo->oidType);
// 					mapOidsBuild[relName] = oid;
// 					expressions::RecordProjection *oidR =
// 							new expressions::RecordProjection(exprR, *oid);
// 					// oidR->registerAs(exprR->getRegisteredRelName(), exprR->getRegisteredAttrName());
// 					//Added in 'wanted expressions'
// 					exprBuild.insert(exprBuild.begin(),oidR);
// 					cout << "Injecting build OID for " << relName << endl;
// 					outputModesBuild.insert(outputModesBuild.begin(), EAGER);
// 				}
// 			}

// 			const rapidjson::Value& build_exprsJSON = val["build_e"];
// 			for (SizeType i = 0; i < build_exprsJSON.Size(); i++){
// 				assert(build_exprsJSON[i].HasMember("e"     ));
// 				assert(build_exprsJSON[i].HasMember("packet"));
// 				assert(build_exprsJSON[i]["packet"].IsInt());
// 				assert(build_exprsJSON[i].HasMember("offset"));
// 				assert(build_exprsJSON[i]["offset"].IsInt());
// 				exprBuild.emplace_back(parseExpression(build_exprsJSON[i]["e"]));
				
// 				expressions::Expression * exprR = exprBuild.back();

// 				outputModesBuild.insert(outputModesBuild.begin(), EAGER);

// 				expressions::RecordProjection *projBuild =
// 						dynamic_cast<expressions::RecordProjection *>(exprR);
// 				if(projBuild == NULL)
// 				{
// 					string error_msg = string(
// 							"[Join: ] Cannot cast to rec projection. Original: ")
// 							+ exprR->getExpressionType()->getType();
// 					LOG(ERROR)<< error_msg;
// 					throw runtime_error(string(error_msg));
// 				}

// 				//Added in 'wanted fields'
// 				RecordAttribute *recAttr = new RecordAttribute(projBuild->getAttribute());
// 				fieldsBuild.push_back(new RecordAttribute(exprR->getRegisteredAs()));

// 				string relName = recAttr->getRelationName();
// 				if (mapOidsBuild.find(relName) == mapOidsBuild.end()) {
// 					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
// 							relName);
// 					RecordAttribute *oid = new RecordAttribute(
// 							recAttr->getRelationName(), activeLoop,
// 							datasetInfo->oidType);
// 					mapOidsBuild[relName] = oid;
// 					expressions::RecordProjection *oidR =
// 							new expressions::RecordProjection(exprR, *oid);
// 					// oidR->registerAs(exprR->getRegisteredRelName(), exprR->getRegisteredAttrName());
// 					//Added in 'wanted expressions'
// 					exprBuild.insert(exprBuild.begin(),oidR);
// 					cout << "Injecting build OID for " << relName << endl;
// 					outputModesBuild.insert(outputModesBuild.begin(), EAGER);
// 				}
// 			}
// 			vector<RecordAttribute*> oidsBuild;
// 			MapToVec(mapOidsBuild, oidsBuild);
// 			Materializer* matBuild = new Materializer(fieldsBuild, exprBuild,
// 					oidsBuild, outputModesBuild);

// 			/*
// 			 * parse output expressions
// 			 * XXX Careful: Assuming numerous output expressions!
// 			 */
// 			assert(val.HasMember("probe_e"));
// 			assert(val["probe_e"].IsArray());
// 			vector<expressions::Expression *> exprProbe   ;
// 			vector<RecordAttribute         *> fieldsProbe ;
// 			map<string, RecordAttribute    *> mapOidsProbe;
// 			vector<materialization_mode>      outputModesProbe;

// 			{
// 				exprProbe.emplace_back(probe_key_expr);
				
// 				expressions::Expression * exprR = exprProbe.back();

// 				outputModesProbe.insert(outputModesProbe.begin(), EAGER);

// 				expressions::RecordProjection *projProbe =
// 						dynamic_cast<expressions::RecordProjection *>(exprR);
// 				if(projProbe == NULL)
// 				{
// 					string error_msg = string(
// 							"[Join: ] Cannot cast to rec projection. Original: ")
// 							+ exprR->getExpressionType()->getType();
// 					LOG(ERROR)<< error_msg;
// 					throw runtime_error(string(error_msg));
// 				}

// 				//Added in 'wanted fields'
// 				RecordAttribute *recAttr = new RecordAttribute(projProbe->getAttribute());
// 				fieldsProbe.push_back(new RecordAttribute(exprR->getRegisteredAs()));

// 				string relName = recAttr->getRelationName();
// 				std::cout << "relName" << " " << relName << std::endl;
// 				if (mapOidsProbe.find(relName) == mapOidsProbe.end()) {
// 					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
// 							relName);
// 					RecordAttribute *oid = new RecordAttribute(
// 							recAttr->getRelationName(), activeLoop,
// 							datasetInfo->oidType);
// 					mapOidsProbe[relName] = oid;
// 					expressions::RecordProjection *oidR =
// 							new expressions::RecordProjection(exprR, *oid);
// 					// oidR->registerAs(exprR->getRegisteredRelName(), exprR->getRegisteredAttrName());
// 					//Added in 'wanted expressions'
// 					exprProbe.insert(exprProbe.begin(),oidR);
// 					cout << "Injecting probe OID for " << relName << endl;
// 					outputModesProbe.insert(outputModesProbe.begin(), EAGER);
// 				}
// 			}

// 			const rapidjson::Value& probe_exprsJSON = val["probe_e"];
// 			for (SizeType i = 0; i < probe_exprsJSON.Size(); i++){
// 				assert(probe_exprsJSON[i].HasMember("e"     ));
// 				assert(probe_exprsJSON[i].HasMember("packet"));
// 				assert(probe_exprsJSON[i]["packet"].IsInt());
// 				assert(probe_exprsJSON[i].HasMember("offset"));
// 				assert(probe_exprsJSON[i]["offset"].IsInt());
// 				exprProbe.emplace_back(parseExpression(probe_exprsJSON[i]["e"]));
				
// 				expressions::Expression * exprR = exprProbe.back();

// 				outputModesProbe.insert(outputModesProbe.begin(), EAGER);

// 				expressions::RecordProjection *projProbe =
// 						dynamic_cast<expressions::RecordProjection *>(exprR);
// 				if(projProbe == NULL)
// 				{
// 					string error_msg = string(
// 							"[Join: ] Cannot cast to rec projection. Original: ")
// 							+ exprR->getExpressionType()->getType();
// 					LOG(ERROR)<< error_msg;
// 					throw runtime_error(string(error_msg));
// 				}

// 				//Added in 'wanted fields'
// 				RecordAttribute *recAttr = new RecordAttribute(projProbe->getAttribute());
// 				fieldsProbe.push_back(new RecordAttribute(exprR->getRegisteredAs()));

// 				string relName = recAttr->getRelationName();
// 				std::cout << "relName" << " " << relName << std::endl;
// 				if (mapOidsProbe.find(relName) == mapOidsProbe.end()) {
// 					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
// 							relName);
// 					RecordAttribute *oid = new RecordAttribute(
// 							recAttr->getRelationName(), activeLoop,
// 							datasetInfo->oidType);
// 					mapOidsProbe[relName] = oid;
// 					expressions::RecordProjection *oidR =
// 							new expressions::RecordProjection(exprR, *oid);
// 					// oidR->registerAs(exprR->getRegisteredRelName(), exprR->getRegisteredAttrName());
// 					//Added in 'wanted expressions'
// 					exprProbe.insert(exprProbe.begin(),oidR);
// 					cout << "Injecting probe OID for " << relName << endl;
// 					outputModesProbe.insert(outputModesProbe.begin(), EAGER);
// 				}
// 			}
// 			vector<RecordAttribute*> oidsProbe;
// 			MapToVec(mapOidsProbe, oidsProbe);
// 			Materializer* matProbe = new Materializer(fieldsProbe, exprProbe,
// 					oidsProbe, outputModesProbe);

// 			newOp = new RadixJoin(predExpr, build_op, probe_op, this->ctx, "radixHashJoin", *matBuild, *matProbe);
// #ifndef NCUDA
// 		}
// #endif
		build_op->setParent(newOp);
		probe_op->setParent(newOp);
	}
	else if(strcmp(opName, "join") == 0)	{
#ifndef NCUDA
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			/* parse operator input */
			RawOperator* build_op = parseOperator(val["build_input"]);
			/* parse operator input */
			RawOperator* probe_op = parseOperator(val["probe_input"]);

			assert(val.HasMember("build_w"));
			assert(val["build_w"].IsArray());
			vector<size_t> build_widths;

			const rapidjson::Value& build_wJSON = val["build_w"];
			for (SizeType i = 0; i < build_wJSON.Size(); i++){
				assert(build_wJSON[i].IsInt());
				build_widths.push_back(build_wJSON[i].GetInt());
			}

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("build_e"));
			assert(val["build_e"].IsArray());
			vector<GpuMatExpr> build_e;
			const rapidjson::Value& build_exprsJSON = val["build_e"];
			for (SizeType i = 0; i < build_exprsJSON.Size(); i++){
				assert(build_exprsJSON[i].HasMember("e"     ));
				assert(build_exprsJSON[i].HasMember("packet"));
				assert(build_exprsJSON[i]["packet"].IsInt());
				assert(build_exprsJSON[i].HasMember("offset"));
				assert(build_exprsJSON[i]["offset"].IsInt());
				expressions::Expression *outExpr = parseExpression(build_exprsJSON[i]["e"]);

				build_e.emplace_back(outExpr, build_exprsJSON[i]["packet"].GetInt(), build_exprsJSON[i]["offset"].GetInt());
			}

			assert(val.HasMember("probe_w"));
			assert(val["probe_w"].IsArray());
			vector<size_t> probe_widths;

			const rapidjson::Value& probe_wJSON = val["probe_w"];
			for (SizeType i = 0; i < probe_wJSON.Size(); i++){
				assert(probe_wJSON[i].IsInt());
				probe_widths.push_back(probe_wJSON[i].GetInt());
			}

			/*
			 * parse output expressions
			 * XXX Careful: Assuming numerous output expressions!
			 */
			assert(val.HasMember("probe_e"));
			assert(val["probe_e"].IsArray());
			vector<GpuMatExpr> probe_e;
			const rapidjson::Value& probe_exprsJSON = val["probe_e"];
			for (SizeType i = 0; i < probe_exprsJSON.Size(); i++){
				assert(probe_exprsJSON[i].HasMember("e"     ));
				assert(probe_exprsJSON[i].HasMember("packet"));
				assert(probe_exprsJSON[i]["packet"].IsInt());
				assert(probe_exprsJSON[i].HasMember("offset"));
				assert(probe_exprsJSON[i]["offset"].IsInt());
				expressions::Expression *outExpr = parseExpression(probe_exprsJSON[i]["e"]);

				probe_e.emplace_back(outExpr, probe_exprsJSON[i]["packet"].GetInt(), probe_exprsJSON[i]["offset"].GetInt());
			}

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new GpuJoin(build_e, build_widths, build_op, 
								probe_e, probe_widths, probe_op, 
								dynamic_cast<GpuRawContext *>(this->ctx));

			build_op->setParent(newOp);
			probe_op->setParent(newOp);
		} else {
#endif
			const char *keyMatLeft = "leftFields";
			const char *keyMatRight = "rightFields";
			const char *keyPred = "p";

			/* parse operator input */
			RawOperator* leftOp = parseOperator(val["leftInput"]);
			RawOperator* rightOp = parseOperator(val["rightInput"]);

			//Predicate
			assert(val.HasMember(keyPred));
			assert(val[keyPred].IsObject());

			expressions::Expression *predExpr = parseExpression(val[keyPred]);
			expressions::BinaryExpression *pred =
					dynamic_cast<expressions::BinaryExpression*>(predExpr);
			if (predExpr == NULL) {
				string error_msg = string(
						"[JOIN: ] Cannot cast to binary predicate. Original: ")
						+ predExpr->getExpressionType()->getType();
				LOG(ERROR)<< error_msg;
				throw runtime_error(string(error_msg));
			}

			/*
			 * *** WHAT TO MATERIALIZE ***
			 * XXX v0: JSON file contains a list of **RecordProjections**
			 * EXPLICIT OIDs injected by PARSER (not in json file by default)
			 * XXX Eager materialization atm
			 *
			 * XXX Why am I not using minimal constructor for materializer yet, as use cases do?
			 * 	-> Because then I would have to encode the OID type in JSON -> can be messy
			 */

			//LEFT SIDE
			assert(val.HasMember(keyMatLeft));
			assert(val[keyMatLeft].IsArray());
			vector<expressions::Expression*> exprsLeft = vector<expressions::Expression*>();
			map<string,RecordAttribute*> mapOidsLeft = map<string,RecordAttribute*>();
			vector<RecordAttribute*> fieldsLeft = vector<RecordAttribute*>();
			vector<materialization_mode> outputModesLeft;
			for (SizeType i = 0; i < val[keyMatLeft].Size(); i++) {
				expressions::Expression *exprL = parseExpression(val[keyMatLeft][i]);

				exprsLeft.push_back(exprL);
				outputModesLeft.insert(outputModesLeft.begin(), EAGER);

				//XXX STRONG ASSUMPTION: Expression is actually a record projection!
				expressions::RecordProjection *projL =
						dynamic_cast<expressions::RecordProjection *>(exprL);
				if(projL == NULL)
				{
					string error_msg = string(
							"[Join: ] Cannot cast to rec projection. Original: ")
							+ exprL->getExpressionType()->getType();
					LOG(ERROR)<< error_msg;
					throw runtime_error(string(error_msg));
				}
				//Added in 'wanted fields'
				RecordAttribute *recAttr = new RecordAttribute(projL->getAttribute());
				fieldsLeft.push_back(recAttr);

				string relName = recAttr->getRelationName();
				if (mapOidsLeft.find(relName) == mapOidsLeft.end()) {
					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
							relName);
					RecordAttribute *oid = new RecordAttribute(
							recAttr->getRelationName(), activeLoop,
							datasetInfo->oidType);
					mapOidsLeft[relName] = oid;
					expressions::RecordProjection *oidL =
							new expressions::RecordProjection(datasetInfo->oidType,
									projL->getExpr(), *oid);
					//Added in 'wanted expressions'
					cout << "Injecting left OID for " << relName << endl;
					exprsLeft.insert(exprsLeft.begin(),oidL);
					outputModesLeft.insert(outputModesLeft.begin(), EAGER);
				}
			}
			vector<RecordAttribute*> oidsLeft = vector<RecordAttribute*>();
			MapToVec(mapOidsLeft,oidsLeft);
			Materializer* matLeft = new Materializer(fieldsLeft, exprsLeft,
						oidsLeft, outputModesLeft);

			//RIGHT SIDE
			assert(val.HasMember(keyMatRight));
			assert(val[keyMatRight].IsArray());
			vector<expressions::Expression*> exprsRight = vector<
					expressions::Expression*>();
			map<string, RecordAttribute*> mapOidsRight = map<string,
					RecordAttribute*>();
			vector<RecordAttribute*> fieldsRight = vector<RecordAttribute*>();
			vector<materialization_mode> outputModesRight;
			for (SizeType i = 0; i < val[keyMatRight].Size(); i++) {
				expressions::Expression *exprR = parseExpression(
						val[keyMatRight][i]);

				exprsRight.push_back(exprR);
				outputModesRight.insert(outputModesRight.begin(), EAGER);

				//XXX STRONG ASSUMPTION: Expression is actually a record projection!
				expressions::RecordProjection *projR =
						dynamic_cast<expressions::RecordProjection *>(exprR);
				if (projR == NULL) {
					string error_msg = string(
							"[Join: ] Cannot cast to rec projection. Original: ")
							+ exprR->getExpressionType()->getType();
					LOG(ERROR)<< error_msg;
					throw runtime_error(string(error_msg));
				}

				//Added in 'wanted fields'
				RecordAttribute *recAttr = new RecordAttribute(
						projR->getAttribute());
				fieldsRight.push_back(recAttr);

				string relName = recAttr->getRelationName();
				if (mapOidsRight.find(relName) == mapOidsRight.end()) {
					InputInfo *datasetInfo = (this->catalogParser).getInputInfo(
							relName);
					RecordAttribute *oid = new RecordAttribute(
							recAttr->getRelationName(), activeLoop,
							datasetInfo->oidType);
					mapOidsRight[relName] = oid;
					expressions::RecordProjection *oidR =
							new expressions::RecordProjection(datasetInfo->oidType,
									projR->getExpr(), *oid);
					//Added in 'wanted expressions'
					exprsRight.insert(exprsRight.begin(),oidR);
					cout << "Injecting right OID for " << relName << endl;
					outputModesRight.insert(outputModesRight.begin(), EAGER);
				}
			}
			vector<RecordAttribute*> oidsRight = vector<RecordAttribute*>();
			MapToVec(mapOidsRight, oidsRight);
			Materializer* matRight = new Materializer(fieldsRight, exprsRight,
					oidsRight, outputModesRight);

			newOp = new RadixJoin(pred,leftOp,rightOp,this->ctx,"radixHashJoin",*matLeft,*matRight);
			leftOp->setParent(newOp);
			rightOp->setParent(newOp);
#ifndef NCUDA
		}
#endif
	}
	else if (strcmp(opName, "nest") == 0) {

		const char *keyGroup = "f";
		const char *keyNull  = "g";
		const char *keyPred  = "p";
		const char *keyExprs = "e";
		const char *keyAccum = "accumulator";
		/* Physical Level Info */
		const char *keyAggrNames = "aggrLabels";
		//Materializer
		const char *keyMat = "fields";

		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		/* get monoid(s) */
		assert(val.HasMember(keyAccum));
		assert(val[keyAccum].IsArray());
		vector<Monoid> accs;
		const rapidjson::Value& accsJSON = val[keyAccum];
		for (SizeType i = 0; i < accsJSON.Size(); i++)
		{
			assert(accsJSON[i].IsString());
			Monoid acc = parseAccumulator(accsJSON[i].GetString());
			accs.push_back(acc);
		}
		/* get label for each of the aggregate values */
		vector<string> aggrLabels;
		assert(val.HasMember(keyAggrNames));
		assert(val[keyAggrNames].IsArray());
		const rapidjson::Value& labelsJSON = val[keyAggrNames];
		for (SizeType i = 0; i < labelsJSON.Size(); i++) {
			assert(labelsJSON[i].IsString());
			aggrLabels.push_back(labelsJSON[i].GetString());
		}

		/* Predicate */
		assert(val.HasMember(keyPred));
		assert(val[keyPred].IsObject());
		expressions::Expression *predExpr = parseExpression(val[keyPred]);

		/* Group By */
		assert(val.HasMember(keyGroup));
		assert(val[keyGroup].IsObject());
		expressions::Expression *groupByExpr = parseExpression(val[keyGroup]);

		/* Null-to-zero Checks */
		//FIXME not used in radix nest yet!
		assert(val.HasMember(keyNull));
		assert(val[keyNull].IsObject());
		expressions::Expression *nullsToZerosExpr = parseExpression(val[keyNull]);

		/* Output aggregate expression(s) */
		assert(val.HasMember(keyExprs));
		assert(val[keyExprs].IsArray());
		vector<expressions::Expression*> outputExprs =
				vector<expressions::Expression*>();
		for (SizeType i = 0; i < val[keyExprs].Size(); i++) {
			expressions::Expression *expr = parseExpression(val[keyExprs][i]);
			outputExprs.push_back(expr);
		}

		/*
		 * *** WHAT TO MATERIALIZE ***
		 * XXX v0: JSON file contains a list of **RecordProjections**
		 * EXPLICIT OIDs injected by PARSER (not in json file by default)
		 * XXX Eager materialization atm
		 *
		 * XXX Why am I not using minimal constructor for materializer yet, as use cases do?
		 * 	-> Because then I would have to encode the OID type in JSON -> can be messy
		 */

		assert(val.HasMember(keyMat));
		assert(val[keyMat].IsArray());
		vector<expressions::Expression*> exprsToMat =
				vector<expressions::Expression*>();
		map<string, RecordAttribute*> mapOids = map<string, RecordAttribute*>();
		vector<RecordAttribute*> fieldsToMat = vector<RecordAttribute*>();
		vector<materialization_mode> outputModes;
		for (SizeType i = 0; i < val[keyMat].Size(); i++) {
			expressions::Expression *expr = parseExpression(val[keyMat][i]);
			exprsToMat.push_back(expr);
			outputModes.insert(outputModes.begin(), EAGER);

			//XXX STRONG ASSUMPTION: Expression is actually a record projection!
			expressions::RecordProjection *proj =
					dynamic_cast<expressions::RecordProjection *>(expr);
			if (proj == NULL) {
				string error_msg = string(
						"[Nest: ] Cannot cast to rec projection. Original: ")
						+ expr->getExpressionType()->getType();
				LOG(ERROR)<< error_msg;
				throw runtime_error(string(error_msg));
			}
			//Added in 'wanted fields'
			RecordAttribute *recAttr =
					new RecordAttribute(proj->getAttribute());
			fieldsToMat.push_back(recAttr);

			string relName = recAttr->getRelationName();
			if (mapOids.find(relName) == mapOids.end()) {
				InputInfo *datasetInfo =
						(this->catalogParser).getInputInfo(relName);
				RecordAttribute *oid =
						new RecordAttribute(recAttr->getRelationName(), activeLoop,
						datasetInfo->oidType);
				mapOids[relName] = oid;
				expressions::RecordProjection *oidProj =
						new expressions::RecordProjection(datasetInfo->oidType,
								proj->getExpr(), *oid);
				//Added in 'wanted expressions'
				LOG(INFO)<< "[Plan Parser: ] Injecting OID for " << relName;
//				std::cout << "[Plan Parser: ] Injecting OID for " << relName << std::endl;
				/* ORDER OF expression fields matters!! OIDs need to be placed first! */
				exprsToMat.insert(exprsToMat.begin(), oidProj);
				outputModes.insert(outputModes.begin(), EAGER);
			}
		}
		vector<RecordAttribute*> oids = vector<RecordAttribute*>();
		MapToVec(mapOids, oids);
		/* FIXME This constructor breaks nest use cases that trigger caching */
		/* Check similar hook in radix-nest.cpp */
		Materializer *matCoarse =
				new Materializer(fieldsToMat, exprsToMat, oids, outputModes);
		
		// Materializer* matCoarse = new Materializer(exprsToMat);

		//Put operator together
		const char *opLabel = "radixNest";
		newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels, predExpr,
				std::vector<expressions::Expression *>{groupByExpr}, nullsToZerosExpr, childOp, opLabel, *matCoarse);
		childOp->setParent(newOp);
	}
	else if (strcmp(opName, "select") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		/* parse filtering expression */
		assert(val.HasMember("p"));
		assert(val["p"].IsObject());
		expressions::Expression *p = parseExpression(val["p"]);

		newOp = new Select(p, childOp);
		childOp->setParent(newOp);
	}
	else if(strcmp(opName,"scan") == 0)	{
		assert(val.HasMember(keyPg));
		assert(val[keyPg].IsObject());
		Plugin *pg = this->parsePlugin(val[keyPg]);

		newOp =  new Scan(this->ctx,*pg);

		GpuColScanPlugin * gpu_scan_pg = dynamic_cast<GpuColScanPlugin *>(pg);
		if (gpu_scan_pg && gpu_scan_pg->getChild()) gpu_scan_pg->getChild()->setParent(newOp);
	} else if(strcmp(opName, "dict-scan") == 0) {
		assert(val.HasMember("relName"));
		assert(val["relName"].IsString());
		auto relName = val["relName"].GetString();

		assert(val.HasMember("attrName"));
		assert(val["attrName"].IsString());
		auto attrName = val["attrName"].GetString();

		assert(val.HasMember("regex"));
		assert(val["regex"].IsString());
		auto regex = val["regex"].GetString();

		auto dictRelName = relName + std::string{"$dict$"} + attrName;

		void * dict = StorageManager::getDictionaryOf(relName + std::string{"."} + attrName);

		InputInfo * datasetInfo = (this->catalogParser).getOrCreateInputInfo(dictRelName);
		RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};
		RecordAttribute * reg_as = new RecordAttribute(dictRelName, attrName, new DStringType(dict)); 
		std::cout << "Registered: " << reg_as->getRelationName() << "." << reg_as->getAttrName() << std::endl;
		rec->appendAttribute(reg_as);

		datasetInfo->exprType = new BagType{*rec};

		newOp =  new DictScan(this->ctx, RecordAttribute{relName, attrName, new DStringType(dict)}, regex, *reg_as);
#ifndef NCUDA
	} else if(strcmp(opName,"cpu-to-gpu") == 0)	{
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new CpuToGpu(childOp, ((GpuRawContext *) this->ctx), projections);
		childOp->setParent(newOp);
#endif
	} else if(strcmp(opName,"block-to-tuples") == 0)	{
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		gran_t granularity = gran_t::GRID;
		bool gpu = true;
		if (val.HasMember("gpu")){
			assert(val["gpu"].IsBool());
			gpu = val["gpu"].GetBool();
			if (!gpu) granularity = gran_t::THREAD;
		}

		if (val.HasMember("granularity")){
			assert(val["granularity"].IsString());
			std::string gran = val["granularity"].GetString();
			std::transform(gran.begin(), gran.end(), gran.begin(), [](unsigned char c){ return std::tolower(c); });
			if      (gran == "grid"  ) granularity = gran_t::GRID;
			else if (gran == "block" ) granularity = gran_t::BLOCK;
			else if (gran == "thread") granularity = gran_t::THREAD;
			else 	assert(false && "granularity must be one of GRID, BLOCK, THREAD");
		}

		vector<expressions::Expression *> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++) {
			assert(val["projections"][i].IsObject());
			projections.push_back(this->parseExpression(val["projections"][i]));
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new BlockToTuples(childOp, ((GpuRawContext *) this->ctx), projections, gpu, granularity);
		childOp->setParent(newOp);
#ifndef NCUDA
	} else if(strcmp(opName,"gpu-to-cpu") == 0)	{
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}


		assert(val.HasMember("queueSize"));
		assert(val["queueSize"].IsInt());
		int size = val["queueSize"].GetInt();

		assert(val.HasMember("granularity"));
		assert(val["granularity"].IsString());
		std::string gran = val["granularity"].GetString();
		std::transform(gran.begin(), gran.end(), gran.begin(), [](unsigned char c){ return std::tolower(c); });
		gran_t g = gran_t::GRID;
		if      (gran == "grid"  ) g = gran_t::GRID;
		else if (gran == "block" ) g = gran_t::BLOCK;
		else if (gran == "thread") g = gran_t::THREAD;
		else 	assert(false && "granularity must be one of GRID, BLOCK, THREAD");

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new GpuToCpu(childOp, ((GpuRawContext *) this->ctx), projections, size, g);
		childOp->setParent(newOp);
#endif
	} else if(strcmp(opName, "tuples-to-block") == 0) {
		bool gpu = false;
		if (val.HasMember("gpu")){
			assert(val["gpu"].IsBool());
			gpu = val["gpu"].GetBool();
		}

		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		int numOfBuckets = 1;
		expressions::Expression *hashExpr = new expressions::IntConstant(0);

		vector<expressions::Expression *> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++){
			assert(val["projections"][i].IsObject());
			projections.push_back(this->parseExpression(val["projections"][i]));
		}
		
		assert(dynamic_cast<GpuRawContext *>(this->ctx));
#ifndef NCUDA
		if (gpu){

			newOp =  new GpuHashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr);
		} else {
#endif
			newOp =  new HashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr);
#ifndef NCUDA
		}
#endif
		childOp->setParent(newOp);
	} else if(strcmp(opName,"hash-rearrange") == 0)	{
		bool gpu = false;
		if (val.HasMember("gpu")){
			assert(val["gpu"].IsBool());
			gpu = val["gpu"].GetBool();
		}
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		int numOfBuckets = 2;
		if (val.HasMember("buckets")){
			assert(val["buckets"].IsInt());
			numOfBuckets = val["buckets"].GetInt();
		}

		RecordAttribute * hashAttr = NULL;
		// register hash as an attribute
		if (val.HasMember("hashProject")){
			assert(val["hashProject"].IsObject());

			hashAttr = parseRecordAttr(val["hashProject"]);

			InputInfo * datasetInfo = (this->catalogParser).getInputInfo(hashAttr->getRelationName());
			RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};

			rec->appendAttribute(hashAttr);

			datasetInfo->exprType = new BagType{*rec};
		}

		assert(val.HasMember("e"));
		assert(val["e"].IsObject());

		expressions::Expression *hashExpr = parseExpression(val["e"]);

		vector<expressions::Expression *> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++){
			assert(val["projections"][i].IsObject());
			projections.push_back(this->parseExpression(val["projections"][i]));
		}
		
		assert(dynamic_cast<GpuRawContext *>(this->ctx));

#ifndef NCUDA
		if (gpu){
			newOp =  new GpuHashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr, hashAttr);
		} else {
#endif
			newOp =  new HashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr, hashAttr);
#ifndef NCUDA
		}
#endif
		childOp->setParent(newOp);
	} else if(strcmp(opName,"mem-move-device") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		bool to_cpu = false;
		if (val.HasMember("to_cpu")){
			assert(val["to_cpu"].IsBool());
			to_cpu = val["to_cpu"].GetBool();
		}

		int slack = 8;
		if (val.HasMember("slack")){
			assert(val["slack"].IsInt());
			slack = val["slack"].GetInt();
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new MemMoveDevice(childOp, ((GpuRawContext *) this->ctx), projections, slack, to_cpu);
		childOp->setParent(newOp);
	} else if(strcmp(opName,"mem-broadcast-device") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		bool to_cpu = false;
		if (val.HasMember("to_cpu")){
			assert(val["to_cpu"].IsBool());
			to_cpu = val["to_cpu"].GetBool();
		}

		int num_of_targets = 1;
		if (val.HasMember("num_of_targets")){
			assert(val["num_of_targets"].IsInt());
			num_of_targets = val["num_of_targets"].GetInt();
		}

		bool always_share = false;
		if (val.HasMember("always_share")){
			assert(val["always_share"].IsBool());
			always_share = val["always_share"].GetBool();
		}


		std::string relName = projections[0]->getRelationName();

		InputInfo * datasetInfo = (this->catalogParser).getOrCreateInputInfo(relName);
		RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};
		RecordAttribute * reg_as = new RecordAttribute(relName, "__broadcastTarget", new IntType()); 
		std::cout << "Registered: " << reg_as->getRelationName() << "." << reg_as->getAttrName() << std::endl;
		rec->appendAttribute(reg_as);

		datasetInfo->exprType = new BagType{*rec};
		
		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new MemBroadcastDevice(childOp, ((GpuRawContext *) this->ctx), projections, num_of_targets, to_cpu, always_share);
		childOp->setParent(newOp);
	} else if(strcmp(opName,"mem-move-local-to") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		int slack = 8;
		if (val.HasMember("slack")){
			assert(val["slack"].IsInt());
			slack = val["slack"].GetInt();
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new MemMoveLocalTo(childOp, ((GpuRawContext *) this->ctx), projections, slack);
		childOp->setParent(newOp);
	} else if(strcmp(opName,"exchange") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++)
		{
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		assert(val.HasMember("numOfParents"));
		assert(val["numOfParents"].IsInt());
		int numOfParents = val["numOfParents"].GetInt();

		int slack = 8;
		if (val.HasMember("slack")){
			assert(val["slack"].IsInt());
			slack = val["slack"].GetInt();
		}

		int producers = 1;
		if (val.HasMember("producers")){
			assert(val["producers"].IsInt());
			producers = val["producers"].GetInt();
		}

		bool numa_local = true;
		bool rand_local_cpu = false;
		expressions::Expression * hash = NULL;
		if (val.HasMember("target")){
			assert(val["target"].IsObject());
			hash = parseExpression(val["target"]);
			numa_local = false;
		}

		if (val.HasMember("rand_local_cpu")){
			assert(hash == NULL && "Can not have both flags set");
			assert(val["rand_local_cpu"].IsBool());
			rand_local_cpu = val["rand_local_cpu"].GetBool();
			numa_local = false;
		}

		if (val.HasMember("numa_local")){
			assert(hash == NULL && "Can not have both flags set");
			assert(!rand_local_cpu);
			assert(numa_local);
			assert(val["numa_local"].IsBool());
			numa_local = val["numa_local"].GetBool();
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new Exchange(childOp, ((GpuRawContext *) this->ctx), numOfParents, projections, slack, hash, numa_local, rand_local_cpu, producers);
		childOp->setParent(newOp);
	} else if(strcmp(opName,"union-all") == 0) {
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsArray());
		std::vector<RawOperator *> children;
		for (SizeType i = 0; i < val["input"].Size(); ++i){
			assert(val["input"][i].IsObject());
			children.push_back(parseOperator(val["input"][i]));
		}

		assert(val.HasMember("projections"));
		assert(val["projections"].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val["projections"].Size(); i++){
			assert(val["projections"][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
			projections.push_back(recAttr);
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new UnionAll(children, ((GpuRawContext *) this->ctx), projections);
		for (const auto &childOp: children) childOp->setParent(newOp);
	} else if(strcmp(opName,"split") == 0) {
		assert(val.HasMember("split_id"));
		assert(val["split_id"].IsInt());
		size_t split_id = val["split_id"].GetInt();

		if (splitOps.count(split_id) == 0){
			/* parse operator input */
			assert(val.HasMember("input"));
			assert(val["input"].IsObject());
			RawOperator* childOp = parseOperator(val["input"]);

			assert(val.HasMember("numOfParents"));
			assert(val["numOfParents"].IsInt());
			int numOfParents = val["numOfParents"].GetInt();

			assert(val.HasMember("projections"));
			assert(val["projections"].IsArray());

			vector<RecordAttribute*> projections;
			for (SizeType i = 0; i < val["projections"].Size(); i++){
				assert(val["projections"][i].IsObject());
				RecordAttribute *recAttr = this->parseRecordAttr(val["projections"][i]);
				projections.push_back(recAttr);
			}

			int slack = 8;
			if (val.HasMember("slack")){
				assert(val["slack"].IsInt());
				slack = val["slack"].GetInt();
			}

			//Does it make sense to have anything rather than rand local ?
			bool numa_local = false; // = true;
			bool rand_local_cpu = false;
			expressions::Expression * hash = NULL;
			if (val.HasMember("target")){
				assert(val["target"].IsObject());
				hash = parseExpression(val["target"]);
				numa_local = false;
			}

			if (val.HasMember("rand_local_cpu")){
				assert(hash == NULL && "Can not have both flags set");
				assert(val["rand_local_cpu"].IsBool());
				rand_local_cpu = val["rand_local_cpu"].GetBool();
				numa_local = false;
			}

			if (val.HasMember("numa_local")){
				assert(hash == NULL && "Can not have both flags set");
				assert(!rand_local_cpu);
				assert(numa_local);
				assert(val["numa_local"].IsBool());
				numa_local = val["numa_local"].GetBool();
			}

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new Split(childOp, ((GpuRawContext *) this->ctx), numOfParents, projections, slack, hash, numa_local, rand_local_cpu);
			splitOps[split_id] = newOp;
			childOp->setParent(newOp);
		} else {
			newOp = splitOps[split_id];
		}
#ifndef NCUDA
	} else if (strcmp(opName, "materializer") == 0){
		/* parse operator input */
		assert(val.HasMember("input"));
		assert(val["input"].IsObject());
		RawOperator* childOp = parseOperator(val["input"]);

		assert(val.HasMember("w"));
		assert(val["w"].IsArray());
		vector<size_t> widths;

		const rapidjson::Value& wJSON = val["w"];
		for (SizeType i = 0; i < wJSON.Size(); i++){
			assert(wJSON[i].IsInt());
			widths.push_back(wJSON[i].GetInt());
		}

		/*
		 * parse output expressions
		 * XXX Careful: Assuming numerous output expressions!
		 */
		assert(val.HasMember("e"));
		assert(val["e"].IsArray());
		vector<GpuMatExpr> e;
		const rapidjson::Value& exprsJSON = val["e"];
		for (SizeType i = 0; i < exprsJSON.Size(); i++){
			assert(exprsJSON[i].HasMember("e"     ));
			assert(exprsJSON[i].HasMember("packet"));
			assert(exprsJSON[i]["packet"].IsInt());
			assert(exprsJSON[i].HasMember("offset"));
			assert(exprsJSON[i]["offset"].IsInt());
			expressions::Expression *outExpr = parseExpression(exprsJSON[i]["e"]);

			e.emplace_back(outExpr, exprsJSON[i]["packet"].GetInt(), exprsJSON[i]["offset"].GetInt());
		}

		/* 'Multi-reduce' used */
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new GpuExprMaterializer(e, widths, childOp, dynamic_cast<GpuRawContext *>(this->ctx));
		} else {
			assert(false && "Unimplemented");
		}
		childOp->setParent(newOp);
#endif
	}
	else	{
		string err = string("Unknown Operator: ") + opName;
		LOG(ERROR) << err;
		throw runtime_error(err);
	}

	return newOp;
}

inline bool ends_with(std::string const &value, std::string const &ending){
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


int lookupInDictionary(string s, const rapidjson::Value& val){
	assert(val.IsObject());
	assert(val.HasMember("path"));
	assert(val["path"].IsString());

	//Input Path
	const char *nameJSON = val["path"].GetString();
	if (ends_with(nameJSON, ".dict")){
		ifstream is(nameJSON);
		string str;
		string prefix = s + ":";
		while(getline(is, str)){
			if (strncmp(str.c_str(), prefix.c_str(), prefix.size()) == 0){
				string encoding{str.c_str() + prefix.size()};
				try {
					size_t pos;
					int enc = stoi(encoding, &pos);
					if (pos + prefix.size() == str.size()) return enc;
					const char *err = "encoded value has extra characters";
					LOG(ERROR)<< err;
					throw runtime_error(err);
				} catch (const std::invalid_argument &){
					const char *err = "invalid dict encoding";
					LOG(ERROR)<< err;
					throw runtime_error(err);
				} catch (const std::out_of_range &){
					const char *err = "out of range dict encoding";
					LOG(ERROR)<< err;
					throw runtime_error(err);
				}
			}
		}
		return -1;// FIXME: this is wrong, we need a binary search, otherwise it breaks ordering
	} else {
		//Prepare Input
		struct stat statbuf;
		stat(nameJSON, &statbuf);
		size_t fsize = statbuf.st_size;

		int fd = open(nameJSON, O_RDONLY);
		if (fd == -1) {
			throw runtime_error(string("json.dict.open"));
		}

		const char *bufJSON = (const char*) mmap(NULL, fsize, PROT_READ,
				MAP_PRIVATE, fd, 0);
		if (bufJSON == MAP_FAILED ) {
			const char *err = "json.dict.mmap";
			LOG(ERROR)<< err;
			throw runtime_error(err);
		}

		Document document; // Default template parameter uses UTF8 and MemoryPoolAllocator.
		if (document.Parse(bufJSON).HasParseError()) {
			const char *err = (string("[CatalogParser: ] Error parsing dictionary ") + string(val["path"].GetString())).c_str();
			LOG(ERROR)<< err;
			throw runtime_error(err);
		}

		assert(document.IsObject());

		if (!document.HasMember(s.c_str())) return -1;// FIXME: this is wrong, we need a binary search, otherwise it breaks ordering

		assert(document[s.c_str()].IsInt());
		return document[s.c_str()].GetInt();
	}
}



expressions::extract_unit ExpressionParser::parseUnitRange(std::string range, RawContext * ctx) {
	if (range == "YEAR" 		) return expressions::extract_unit::YEAR;
	if (range == "MONTH" 		) return expressions::extract_unit::MONTH;
	if (range == "DAY" 			) return expressions::extract_unit::DAYOFMONTH;
	if (range == "HOUR" 		) return expressions::extract_unit::HOUR;
	if (range == "MINUTE" 		) return expressions::extract_unit::MINUTE;
	if (range == "SECOND" 		) return expressions::extract_unit::SECOND;
	if (range == "QUARTER" 		) return expressions::extract_unit::QUARTER;
	if (range == "WEEK" 		) return expressions::extract_unit::WEEK;
	if (range == "MILLISECOND" 	) return expressions::extract_unit::MILLISECOND;
	if (range == "DOW" 			) return expressions::extract_unit::DAYOFWEEK;
	if (range == "DOY" 			) return expressions::extract_unit::DAYOFYEAR;
	if (range == "DECADE" 		) return expressions::extract_unit::DECADE;
	if (range == "CENTURY" 		) return expressions::extract_unit::CENTURY;
	if (range == "MILLENNIUM" 	) return expressions::extract_unit::MILLENNIUM;
	// case "YEAR_TO_MONTH" 	:
	// case "DAY_TO_HOUR" 		:
	// case "DAY_TO_MINUTE" 	:
	// case "DAY_TO_SECOND" 	:
	// case "HOUR_TO_MINUTE" 	:
	// case "HOUR_TO_SECOND" 	:
	// case "MINUTE_TO_SECOND" :
	// case "EPOCH" 			:
	// default:{
	string err = string("Unsupoport TimeUnitRange: ") + range;
	LOG(ERROR)<< err;
	throw runtime_error(err);
	// }
}


/*
 *	enum ExpressionId	{ CONSTANT, ARGUMENT, RECORD_PROJECTION, RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE };
 *	FIXME / TODO No Merge yet!! Will be needed for parallelism!
 *	TODO Add NotExpression ?
 */

expressions::Expression* ExpressionParser::parseExpression(const rapidjson::Value& val, RawContext * ctx) {

	const char *keyExpression = "expression";
	const char *keyArgNo = "argNo";
	const char *keyExprType = "type";

	/* Input Argument specifics */
	const char *keyAtts = "attributes";

	/* Record Projection specifics */
	const char *keyInnerExpr = "e";
	const char *keyProjectedAttr = "attribute";

	/* Record Construction specifics */
	const char *keyAttsConstruction = "attributes";
	const char *keyAttrName = "name";
	const char *keyAttrExpr = "e";

	/* If-else specifics */
	const char *keyCond = "cond";
	const char *keyThen = "then";
	const char *keyElse = "else";

	/*Binary operator(s) specifics */
	const char *leftArg = "left";
	const char *rightArg = "right";

	assert(val.HasMember(keyExpression));
	assert(val[keyExpression].IsString());
	const char *valExpression = val[keyExpression].GetString();

	expressions::Expression* retValue = NULL;

	assert(!val.HasMember("isNull") || val["isNull"].IsBool());
	bool isNull = val.HasMember("isNull") && val["isNull"].GetBool();

	const auto &createNull=[&](ExpressionType * b){
		RawValue rv{
			UndefValue::get(b->getLLVMType(ctx->getLLVMContext())),
			ctx->createTrue()
		};

		return new expressions::RawValueExpression(b, rv);
	};

	if (strcmp(valExpression, "bool") == 0) {
		if (isNull) {
			retValue = createNull(new BoolType());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsBool());
			retValue = new expressions::BoolConstant(val["v"].GetBool());
		}
	} else if (strcmp(valExpression, "int") == 0) {
		if (isNull) {
			retValue = createNull(new IntType());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsInt());
			retValue = new expressions::IntConstant(val["v"].GetInt());
		}
	} else if (strcmp(valExpression, "int64") == 0) {
		if (isNull) {
			retValue = createNull(new Int64Type());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsInt64());
			retValue = new expressions::Int64Constant(val["v"].GetInt64());
		}
	} else if (strcmp(valExpression, "float") == 0) {
		if (isNull) {
			retValue = createNull(new FloatType());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsDouble());
			retValue = new expressions::FloatConstant(val["v"].GetDouble());
		}
	} else if (strcmp(valExpression, "date") == 0) {
		if (isNull) {
			retValue = createNull(new DateType());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsInt64());
			retValue = new expressions::DateConstant(val["v"].GetInt64());
		}
	} else if (strcmp(valExpression, "string") == 0) {
		if (isNull) {
			retValue = createNull(new StringType());
		} else {
			assert(val.HasMember("v"));
			assert(val["v"].IsString());
			string *stringVal = new string(val["v"].GetString());
			retValue = new expressions::StringConstant(*stringVal);
		}
	} else if (strcmp(valExpression, "dstring") == 0) { //FIMXE: do something better, include the dictionary
		if (isNull) {
			retValue = createNull(new DStringType());
		} else {
			assert(val.HasMember("v"));
			if (val["v"].IsInt()){
				retValue = new expressions::IntConstant(val["v"].GetInt());
			} else {
				assert(val["v"].IsString());
				assert(val.HasMember("dict"));

				int sVal = lookupInDictionary(val["v"].GetString(), val["dict"]);
				std::cout << sVal << " " << val["v"].GetString() << std::endl;
				retValue = new expressions::IntConstant(sVal);
			}
		}
	} else if (strcmp(valExpression, "argument") == 0) {
		assert(!isNull);
		/* exprType */
		assert(val.HasMember(keyExprType));
		assert(val[keyExprType].IsObject());
		ExpressionType *exprType = parseExpressionType(val[keyExprType]);

		/* argNo */
		assert(val.HasMember(keyArgNo));
		assert(val[keyArgNo].IsInt());
		int argNo = val[keyArgNo].GetInt();

		/* 'projections' / attributes */
		assert(val.HasMember(keyAtts));
		assert(val[keyAtts].IsArray());

		list<RecordAttribute> atts = list<RecordAttribute>();
		const rapidjson::Value& attributes = val[keyAtts]; // Using a reference for consecutive access is handy and faster.
		for (SizeType i = 0; i < attributes.Size(); i++) // rapidjson uses SizeType instead of size_t.
		{
			RecordAttribute *recAttr = parseRecordAttr(attributes[i]);
			atts.push_back(*recAttr);
		}
		retValue = new expressions::InputArgument(exprType, argNo, atts);

	} else if (strcmp(valExpression, "recordProjection") == 0) {
		assert(!isNull);

		/* e: expression over which projection is calculated */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		/* projected attribute */
		assert(val.HasMember(keyProjectedAttr));
		assert(val[keyProjectedAttr].IsObject());
		RecordAttribute *recAttr = parseRecordAttr(val[keyProjectedAttr]);

		/* exprType */
		if (val.HasMember(keyExprType)){
			assert(val[keyExprType].IsObject());
			ExpressionType * exprType = parseExpressionType(val[keyExprType]);

			if (exprType->getTypeID() != recAttr->getOriginalType()->getTypeID()){
				string err = string("recordProjection type differed from projected attribute's type (") +
								exprType->getType() +
								"!=" +
								recAttr->getOriginalType()->getType() + 
								")";
				LOG(WARNING)<< err;
				cout << err << endl;
			}
			retValue = new expressions::RecordProjection(exprType, expr, *recAttr);
		} else {
			retValue = new expressions::RecordProjection(expr, *recAttr);
		}
	} else if (strcmp(valExpression, "recordConstruction") == 0) {
		assert(!isNull);
		/* exprType */
		// assert(val.HasMember(keyExprType));
		// assert(val[keyExprType].IsObject());
		// ExpressionType *exprType = parseExpressionType(val[keyExprType]);

		/* attribute construction(s) */
		assert(val.HasMember(keyAttsConstruction));
		assert(val[keyAttsConstruction].IsArray());

		list<expressions::AttributeConstruction> *newAtts = new list<expressions::AttributeConstruction>();
		const rapidjson::Value& attributeConstructs = val[keyAttsConstruction]; // Using a reference for consecutive access is handy and faster.
		for (SizeType i = 0; i < attributeConstructs.Size(); i++) // rapidjson uses SizeType instead of size_t.
		{
			assert(attributeConstructs[i].HasMember(keyAttrName));
			assert(attributeConstructs[i][keyAttrName].IsString());
			string newAttrName = attributeConstructs[i][keyAttrName].GetString();

			assert(attributeConstructs[i].HasMember(keyAttrExpr));
			assert(attributeConstructs[i][keyAttrExpr].IsObject());
			expressions::Expression *newAttrExpr = parseExpression(attributeConstructs[i][keyAttrExpr], ctx);

			expressions::AttributeConstruction *newAttr =
					new expressions::AttributeConstruction(newAttrName,newAttrExpr);
			newAtts->push_back(*newAttr);
		}
		retValue = new expressions::RecordConstruction(*newAtts);
	} else if (strcmp(valExpression, "extract") == 0) {
		assert(val.HasMember("unitrange"));
		assert(val["unitrange"].IsString());

		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		auto u = parseUnitRange(val["unitrange"].GetString(), ctx);
		retValue = new expressions::ExtractExpression(expr, u);
	} else if (strcmp(valExpression,"if") == 0)	{
		assert(!isNull);
		/* if cond */
		assert(val.HasMember(keyCond));
		assert(val[keyCond].IsObject());
		expressions::Expression *condExpr = parseExpression(val[keyCond], ctx);

		/* then expression */
		assert(val.HasMember(keyThen));
		assert(val[keyThen].IsObject());
		expressions::Expression *thenExpr = parseExpression(val[keyThen], ctx);

		/* else expression */
		assert(val.HasMember(keyElse));
		assert(val[keyElse].IsObject());
		expressions::Expression *elseExpr = parseExpression(val[keyElse], ctx);

		retValue = new expressions::IfThenElse(condExpr,thenExpr,elseExpr);
	}
	/*
	 * BINARY EXPRESSIONS
	 */
	else if (strcmp(valExpression, "eq") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::EqExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "neq") == 0) {
		assert(!isNull);
		/* left child */
				assert(val.HasMember(leftArg));
				assert(val[leftArg].IsObject());
				expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

				/* right child */
				assert(val.HasMember(rightArg));
				assert(val[rightArg].IsObject());
				expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

				retValue = new expressions::NeExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "lt") == 0) {
		assert(!isNull);
		/* left child */
				assert(val.HasMember(leftArg));
				assert(val[leftArg].IsObject());
				expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

				/* right child */
				assert(val.HasMember(rightArg));
				assert(val[rightArg].IsObject());
				expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

				retValue = new expressions::LtExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "le") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::LeExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "gt") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::GtExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "ge") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::GeExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "and") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::AndExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "or") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::OrExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "add") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		// ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::AddExpression(leftExpr, rightExpr);
	} else if (strcmp(valExpression, "sub") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		// ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::SubExpression(leftExpr, rightExpr);
	} else if (strcmp(valExpression, "neg") == 0) {
		assert(!isNull);
		/* right child */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		retValue = new expressions::NegExpression(expr);
	} else if (strcmp(valExpression, "is_not_null") == 0) {
		assert(!isNull);
		/* right child */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		retValue = new expressions::TestNullExpression(expr, false);
	} else if (strcmp(valExpression, "is_null") == 0) {
		assert(!isNull);
		/* right child */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		retValue = new expressions::TestNullExpression(expr, true);
	} else if (strcmp(valExpression, "cast") == 0) {
		assert(!isNull);
		/* right child */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr], ctx);

		assert(val.HasMember(keyExprType));
		assert(val[keyExprType].IsObject());

		ExpressionType * t = parseExpressionType(val[keyExprType]);

		retValue = new expressions::CastExpression(t, expr);
	} else if (strcmp(valExpression, "multiply") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::MultExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "div") == 0) {
		assert(!isNull);
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg], ctx);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg], ctx);

		retValue = new expressions::DivExpression(leftExpr,rightExpr);
	} else if (strcmp(valExpression, "merge") == 0) {
		assert(!isNull);
		string err = string("(Still) unsupported expression: ") + valExpression;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	} else {
		string err = string("Unknown expression: ") + valExpression;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	if (retValue && val.HasMember("register_as")){
		assert(val["register_as"].IsObject());
		RecordAttribute * reg_as = parseRecordAttr(val["register_as"], retValue->getExpressionType());
		assert(reg_as && "Error registering expression as attribute");

		InputInfo * datasetInfo = (this->catalogParser).getOrCreateInputInfo(reg_as->getRelationName());
		RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};
		std::cout << "Registered: " << reg_as->getRelationName() << "." << reg_as->getAttrName() << std::endl;
		rec->appendAttribute(reg_as);

		datasetInfo->exprType = new BagType{*rec};

		retValue->registerAs(reg_as);
	}

	return retValue;
}


/*
 * enum typeID	{ BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET, INT64, COMPOSITE };
 * FIXME / TODO: Do I need to cater for 'composite' types?
 * IIRC, they only occur as OIDs / complex caches
*/
ExpressionType* ExpressionParser::parseExpressionType(const rapidjson::Value& val) {

	/* upper-level keys */
	const char *keyExprType = "type";
	const char *keyCollectionType = "inner";

	/* Related to record types */
	const char *keyRecordAttrs = "attributes";

	assert(val.HasMember(keyExprType));
	assert(val[keyExprType].IsString());
	const char *valExprType = val[keyExprType].GetString();

	if (strcmp(valExprType, "bool") == 0) {
		return new BoolType();
	} else if (strcmp(valExprType, "int") == 0) {
		return new IntType();
	} else if (strcmp(valExprType, "int64") == 0) {
		return new Int64Type();
	} else if (strcmp(valExprType, "float") == 0) {
		return new FloatType();
	} else if (strcmp(valExprType, "date") == 0) {
		return new DateType();
	} else if (strcmp(valExprType, "string") == 0) {
		return new StringType();
	} else if (strcmp(valExprType, "dstring") == 0) {
		return new DStringType(NULL);
	} else if (strcmp(valExprType, "set") == 0) {
		assert(val.HasMember("inner"));
		assert(val["inner"].IsObject());
		ExpressionType *innerType = parseExpressionType(val["inner"]);
		return new SetType(*innerType);
	} else if (strcmp(valExprType, "bag") == 0) {
		assert(val.HasMember("inner"));
		assert(val["inner"].IsObject());
		ExpressionType *innerType = parseExpressionType(val["inner"]);
		return new BagType(*innerType);
	} else if (strcmp(valExprType, "list") == 0) {
		assert(val.HasMember("inner"));
		assert(val["inner"].IsObject());
		ExpressionType *innerType = parseExpressionType(val["inner"]);
		return new ListType(*innerType);
	} else if (strcmp(valExprType, "record") == 0) {
		if (val.HasMember("attributes")){
			assert(val["attributes"].IsArray());

			list<RecordAttribute*> atts = list<RecordAttribute*>();
			const rapidjson::Value& attributes = val["attributes"]; // Using a reference for consecutive access is handy and faster.
			for (SizeType i = 0; i < attributes.Size(); i++) // rapidjson uses SizeType instead of size_t.
			{
				//Starting from 1
				RecordAttribute *recAttr = parseRecordAttr(attributes[i]);
				atts.push_back(recAttr);
			}
			return new RecordType(atts);
		} else if (val.HasMember("relName")){
			assert(val["relName"].IsString());

			return getRecordType(val["relName"].GetString());
		} else {
			return new RecordType();
		}
	} else if (strcmp(valExprType, "composite") == 0) {
		string err = string("(Still) Unsupported expression type: ")
				+ valExprType;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	} else {
		string err = string("Unknown expression type: ") + valExprType;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

}

RecordType * ExpressionParser::getRecordType(string relName){
	//Lookup in catalog based on name
	InputInfo *datasetInfo = (this->catalogParser).getInputInfoIfKnown(relName);
	if (datasetInfo == NULL) return NULL;

	/* Retrieve RecordType */
	/* Extract inner type of collection */
	CollectionType *collType = dynamic_cast<CollectionType*>(datasetInfo->exprType);
	if(collType == NULL)	{
		string error_msg = string("[Type Parser: ] Cannot cast to collection type. Original intended type: ") + datasetInfo->exprType->getType();
		LOG(ERROR)<< error_msg;
		throw runtime_error(string(error_msg));
	}
	/* For the current plugins, the expression type is unambiguously RecordType */
	const ExpressionType& nestedType = collType->getNestedType();
	const RecordType& recType_ = dynamic_cast<const RecordType&>(nestedType);
	return new RecordType(recType_.getArgs());
}

const RecordAttribute * ExpressionParser::getAttribute(string relName, string attrName){
	RecordType * recType = getRecordType(relName);
	if (recType == NULL) return NULL;

	return recType->getArg(attrName);
}

RecordAttribute* ExpressionParser::parseRecordAttr(const rapidjson::Value& val, const ExpressionType * defaultType) {

	const char *keyRecAttrType = "type";
	const char *keyRelName = "relName";
	const char *keyAttrName = "attrName";
	const char *keyAttrNo = "attrNo";

	assert(val.HasMember(keyRelName));
	assert(val[keyRelName].IsString());
	string relName = val[keyRelName].GetString();

	assert(val.HasMember(keyAttrName));
	assert(val[keyAttrName].IsString());
	string attrName = val[keyAttrName].GetString();

	const RecordAttribute * attr = getAttribute(relName, attrName);

	int attrNo;
	if (val.HasMember(keyAttrNo)){
		assert(val[keyAttrNo].IsInt());
		attrNo = val[keyAttrNo].GetInt();
	} else {
		if (!attr) attrNo = -1;
		else       attrNo = attr->getAttrNo();
	}

	const ExpressionType* recArgType;
	if (val.HasMember(keyRecAttrType)){
		assert(val[keyRecAttrType].IsObject());
		recArgType = parseExpressionType(val[keyRecAttrType]);
	} else {
		if (attr){
			recArgType = attr->getOriginalType();
		} else {
			if (defaultType) recArgType = defaultType;
			else             assert(false && "Attribute not found");
		}
	}

	bool is_block = false;
	if (val.HasMember("isBlock")){
		assert(val["isBlock"].IsBool());
		is_block = val["isBlock"].GetBool();
	}

	return new RecordAttribute(attrNo, relName, attrName, recArgType, is_block);
}

Monoid ExpressionParser::parseAccumulator(const char *acc) {

	if (strcmp(acc, "sum") == 0) {
		return SUM;
	} else if (strcmp(acc, "max") == 0) {
		return MAX;
	} else if (strcmp(acc, "multiply") == 0) {
		return MULTIPLY;
	} else if (strcmp(acc, "or") == 0) {
		return OR;
	} else if (strcmp(acc, "and") == 0) {
		return AND;
	} else if (strcmp(acc, "union") == 0) {
		return UNION;
	} else if (strcmp(acc, "bagunion") == 0) {
		return BAGUNION;
	} else if (strcmp(acc, "append") == 0) {
		return APPEND;
	} else {
		string err = string("Unknown Monoid: ") + acc;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}
}

/**
 * {"name": "foo", "type": "csv", ... }
 * FIXME / TODO If we introduce more plugins, this code must be extended
 */
Plugin* PlanExecutor::parsePlugin(const rapidjson::Value& val)	{

	Plugin *newPg = NULL;

	const char *keyInputName = "name";
	const char *keyPgType = "type";

	/*
	 * CSV-specific
	 */
	//which fields to project
	const char *keyProjectionsCSV = "projections";
	//pm policy
	const char *keyPolicy = "policy";
	//line hint
	const char *keyLineHint = "lines";
	//OPTIONAL: which delimiter to use
	const char *keyDelimiter = "delimiter";
	//OPTIONAL: are string values wrapped in brackets?
	const char *keyBrackets = "brackets";

	/*
	 * BinRow
	 */
	const char *keyProjectionsBinRow = "projections";

	/*
	 * GPU
	 */
	const char *keyProjectionsGPU = "projections";

	/*
	 * BinCol
	 */
	const char *keyProjectionsBinCol = "projections";

	assert(val.HasMember(keyInputName));
	assert(val[keyInputName].IsString());
	string datasetName = val[keyInputName].GetString();

	assert(val.HasMember(keyPgType));
	assert(val[keyPgType].IsString());
	const char *pgType = val[keyPgType].GetString();

	//Lookup in catalog based on name
	InputInfo *datasetInfo = (this->catalogParser).getInputInfoIfKnown(datasetName);
	bool pluginExisted = true;

	if (!datasetInfo){
		RecordType * rec = new RecordType();

		if (val.HasMember("schema")){
			const auto &schema = val["schema"];
			assert(schema.IsArray());

			for (SizeType i = 0; i < schema.Size(); i++)
			{
				assert(schema[i].IsObject());
				RecordAttribute *recAttr = parseRecordAttr(schema[i]);
				
				std::cout << "Plugin Registered: " << 
					recAttr->getRelationName() << "." << 
					recAttr->getAttrName() << std::endl;
				
				rec->appendAttribute(recAttr);
			}
		}

		datasetInfo            = new InputInfo()  ;
		datasetInfo->exprType  = new BagType(*rec);
		datasetInfo->path      = datasetName;

		if (val.HasMember("schema")){
			// Register it to make it visible to the plugin
			datasetInfo->oidType = NULL;
			(this->catalogParser).setInputInfo(datasetName, datasetInfo);
		}

		pluginExisted = false;
	}

	//Dynamic allocation because I have to pass reference later on
	string *pathDynamicCopy = new string(datasetInfo->path);

	/* Retrieve RecordType */
	/* Extract inner type of collection */
	CollectionType *collType = dynamic_cast<CollectionType*>(datasetInfo->exprType);
	if(collType == NULL)	{
		string error_msg = string("[Plugin Parser: ] Cannot cast to collection type. Original intended type: ") + datasetInfo->exprType->getType();
		LOG(ERROR)<< error_msg;
		throw runtime_error(string(error_msg));
	}
	/* For the current plugins, the expression type is unambiguously RecordType */
	const ExpressionType& nestedType = collType->getNestedType();
	const RecordType& recType_ = dynamic_cast<const RecordType&>(nestedType);
	//circumventing the presence of const
	RecordType *recType = new RecordType(recType_.getArgs());

	if (strcmp(pgType, "csv") == 0) {
//		cout<<"Original intended type: " << datasetInfo.exprType->getType()<<endl;
//		cout<<"File path: " << datasetInfo.path<<endl;

		/* Projections come in an array of Record Attributes */
		assert(val.HasMember(keyProjectionsCSV));
		assert(val[keyProjectionsCSV].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val[keyProjectionsCSV].Size(); i++)
		{
			assert(val[keyProjectionsCSV][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val[keyProjectionsCSV][i]);
			projections.push_back(recAttr);
		}

		assert(val.HasMember(keyLineHint));
		assert(val[keyLineHint].IsInt());
		int linehint = val[keyLineHint].GetInt();

		assert(val.HasMember(keyPolicy));
		assert(val[keyPolicy].IsInt());
		int policy = val[keyPolicy].GetInt();

		char delim = ',';
		if (val.HasMember(keyDelimiter)) {
			assert(val[keyDelimiter].IsString());
			delim = (val[keyDelimiter].GetString())[0];
		}
		else
		{
			string err = string("WARNING - NO DELIMITER SPECIFIED. FALLING BACK TO DEFAULT");
			LOG(WARNING)<< err;
			cout << err << endl;
		}

		bool stringBrackets = true;
		if (val.HasMember(keyBrackets)) {
			assert(val[keyBrackets].IsBool());
			stringBrackets = val[keyBrackets].GetBool();
		}

		std::cout << *pathDynamicCopy << std::endl;
		newPg = new pm::CSVPlugin(this->ctx, *pathDynamicCopy, *recType,
				projections, delim, linehint, policy, stringBrackets);
	} else if (strcmp(pgType, "json") == 0) {
		assert(val.HasMember(keyLineHint));
		assert(val[keyLineHint].IsInt());
		int linehint = val[keyLineHint].GetInt();

		newPg = new jsonPipelined::JSONPlugin(this->ctx, *pathDynamicCopy, datasetInfo->exprType);
	} else if (strcmp(pgType, "binrow") == 0) {
		assert(val.HasMember(keyProjectionsBinRow));
		assert(val[keyProjectionsBinRow].IsArray());

		vector<RecordAttribute*> *projections = new vector<RecordAttribute*>();
		for (SizeType i = 0; i < val[keyProjectionsBinRow].Size(); i++) {
			assert(val[keyProjectionsBinRow][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(
					val[keyProjectionsBinRow][i]);
			projections->push_back(recAttr);
		}

		newPg = new BinaryRowPlugin(this->ctx, *pathDynamicCopy, *recType,
				*projections);
	} else if (strcmp(pgType, "bincol") == 0) {
		assert(val.HasMember(keyProjectionsBinCol));
		assert(val[keyProjectionsBinCol].IsArray());

		vector<RecordAttribute*> *projections = new vector<RecordAttribute*>();
		for (SizeType i = 0; i < val[keyProjectionsBinCol].Size(); i++) {
			assert(val[keyProjectionsBinCol][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(
					val[keyProjectionsBinCol][i]);
			projections->push_back(recAttr);
		}

		bool sizeInFile = true;
		if (val.HasMember("sizeInFile")){
			assert(val["sizeInFile"].IsBool());
			sizeInFile = val["sizeInFile"].GetBool();
		}
		newPg = new BinaryColPlugin(this->ctx, *pathDynamicCopy, *recType,
				*projections, sizeInFile);
	} else if (strcmp(pgType, "gpu") == 0) {
		assert(val.HasMember(keyProjectionsGPU));
		assert(val[keyProjectionsGPU].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val[keyProjectionsGPU].Size(); i++)
		{
			assert(val[keyProjectionsGPU][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val[keyProjectionsGPU][i]);
			projections.push_back(recAttr);
		}
		assert(dynamic_cast<GpuRawContext *>(this->ctx));

		RawOperator* childOp = NULL;
		if (val.HasMember("input")) {
			assert(val["input"].IsObject());
			childOp = parseOperator(val["input"]);
		}

		newPg = new GpuColScanPlugin(dynamic_cast<GpuRawContext *>(this->ctx), *pathDynamicCopy, *recType, projections, childOp);

	} else if (strcmp(pgType, "block") == 0) {
		assert(val.HasMember(keyProjectionsGPU));
		assert(val[keyProjectionsGPU].IsArray());

		vector<RecordAttribute*> projections;
		for (SizeType i = 0; i < val[keyProjectionsGPU].Size(); i++)
		{
			assert(val[keyProjectionsGPU][i].IsObject());
			RecordAttribute *recAttr = this->parseRecordAttr(val[keyProjectionsGPU][i]);
			projections.push_back(recAttr);
		}
		assert(dynamic_cast<GpuRawContext *>(this->ctx));

		newPg = new ScanToBlockSMPlugin(dynamic_cast<GpuRawContext *>(this->ctx), *pathDynamicCopy, *recType, projections);
	} else {
		string err = string("Unknown Plugin Type: ") + pgType;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	activePlugins.push_back(newPg);
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.registerPlugin(*pathDynamicCopy,newPg);
	datasetInfo->oidType = newPg->getOIDType();
	(this->catalogParser).setInputInfo(datasetName,datasetInfo);
	return newPg;
}

#include <dirent.h>
#include <stdlib.h>

void CatalogParser::parseCatalogFile(std::string file){
	//key aliases
	const char *keyInputPath = "path";
	const char *keyExprType =  "type";

	//Prepare Input
	struct stat statbuf;
	stat(file.c_str(), &statbuf);
	size_t fsize = statbuf.st_size;

	int fd = open(file.c_str(), O_RDONLY);
	if (fd < 0) {
		std::string err = "failed to open file: " + file;
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}
	const char *bufJSON = (const char*) mmap(NULL, fsize, PROT_READ,
			MAP_PRIVATE, fd, 0);
	if (bufJSON == MAP_FAILED ) {
		std::string err = "json.mmap";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	Document document; // Default template parameter uses UTF8 and MemoryPoolAllocator.
	auto & parsed = document.Parse(bufJSON);
	if (parsed.HasParseError()) {
		ParseResult ok = (ParseResult) parsed;
		fprintf(stderr, "[CatalogParser: ] Error parsing physical plan: %s (%lu)", RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset());
			const char *err = "[CatalogParser: ] Error parsing physical plan";
		LOG(ERROR)<< err << ": " << RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()) << "(" << ok.Offset() << ")";
		throw runtime_error(err);
	}

	//Start plan traversal.
	printf("\nParsing catalog information:\n");
	assert(document.IsObject());

	for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin();
			itr != document.MemberEnd(); ++itr) {
		// printf("Key of member is %s\n", itr->name.GetString());

		assert(itr->value.IsObject());
		assert((itr->value)[keyInputPath].IsString());
		string inputPath = ((itr->value)[keyInputPath]).GetString();
		assert((itr->value)[keyExprType].IsObject());
		ExpressionType *exprType = exprParser.parseExpressionType((itr->value)[keyExprType]);
		InputInfo *info = new InputInfo();
		info->exprType = exprType;
		info->path = inputPath;
		//Initialized by parsePlugin() later on
		info->oidType = NULL;
//			(this->inputs)[itr->name.GetString()] = info;
		(this->inputs)[info->path] = info;

		setInputInfo(info->path, info);
	}
}

void CatalogParser::parseDir(std::string dir){
	//FIXME: we can do that in a portable way with C++17, but for now because we
	// are using libstdc++, upgrading to C++17 and using <filesystem> causes 
	// linking problems in machines with old gcc version
	DIR *d = opendir(dir.c_str());
	if (!d) {
		std::string err = "Failed to open dir: " + dir + " (" + strerror(errno) + ")";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	dirent *entry;
	while ((entry = readdir(d)) != NULL) {
		std::string fname{entry->d_name};

		if (strcmp(entry->d_name, "..") == 0) continue;
		if (strcmp(entry->d_name, "." ) == 0) continue;

		std::string origd{dir + "/" + fname};
		//Use this to canonicalize paths:
		// std::string pathd{realpath(origd.c_str(), NULL)};
		std::string pathd{origd};

		struct stat s;
		stat(pathd.c_str(), &s);

		if (S_ISDIR(s.st_mode)) {
			parseDir(pathd);
		} else if (fname == "catalog.json" && S_ISREG(s.st_mode)){
			parseCatalogFile(pathd);
		} /* else skipping */
	}
	closedir(d);
}


/**
 * {"datasetname": {"path": "foo", "type": { ... } }
 */
CatalogParser::CatalogParser(const char *catalogPath, GpuRawContext *context): exprParser(*this), context(context) {
	parseDir(catalogPath);
}


InputInfo *CatalogParser::getOrCreateInputInfo(string inputName){
	InputInfo * ret = getInputInfoIfKnown(inputName);
	
	if (!ret){
		RecordType * rec = new RecordType();

		ret            = new InputInfo()  ;
		ret->exprType  = new BagType(*rec);
		ret->path      = inputName;

		RawCatalog& catalog = RawCatalog::getInstance();
		
		assert(context && "A GpuRawContext is required to register relationships on the fly");
		vector<RecordAttribute *> projs;
		Plugin * newPg = new pm::CSVPlugin(context, inputName, *rec, projs, ',', 10, 1, false);
		catalog.registerPlugin(*(new string(inputName)), newPg);
		ret->oidType   = newPg->getOIDType();

		setInputInfo(inputName, ret);
	}

	return ret;
}
