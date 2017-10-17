#include "plan/plan-parser.hpp"
#include "plugins/gpu-col-scan-plugin.hpp"
#include "plugins/gpu-col-scan-to-blocks-plugin.hpp"
#include "plugins/scan-to-blocks-sm-plugin.hpp"
#include "operators/gpu/gpu-join.hpp"
#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "operators/gpu/gpu-hash-group-by-chained.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/cpu-to-gpu.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/exchange.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/block-to-tuples.hpp"

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


PlanExecutor::PlanExecutor(const char *planPath, CatalogParser& cat, RawContext * ctx) :
		planPath(planPath), moduleName(""), catalogParser(cat), ctx(ctx), exprParser(cat) {

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
		if (val.HasMember("gpu") && val["gpu"].GetBool()){
			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp = new opt::GpuReduce(accs, e, p, childOp, dynamic_cast<GpuRawContext *>(this->ctx));
		} else {
			newOp = new opt::Reduce(accs, e, p, childOp, this->ctx,true,moduleName);
		}
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
	} else if(strcmp(opName, "hashgroupby-chained") == 0)	{
		/* parse operator input */
		RawOperator* child = parseOperator(val["input"]);

		assert(val.HasMember("gpu") && val["gpu"].GetBool());

		assert(val.HasMember("hash_bits"));
		assert(val["hash_bits"].IsInt());
		int hash_bits = val["hash_bits"].GetInt();

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
		newOp = new GpuHashGroupByChained(e, widths, key_expr, child, hash_bits,
							dynamic_cast<GpuRawContext *>(this->ctx), maxInputSize);

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

		assert(val.HasMember("gpu") && val["gpu"].GetBool());

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

		assert(val.HasMember("build_k"));
		expressions::Expression *build_key_expr = parseExpression(val["build_k"]);

		assert(val.HasMember("probe_w"));
		assert(val["probe_w"].IsArray());
		vector<size_t> probe_widths;

		const rapidjson::Value& probe_wJSON = val["probe_w"];
		for (SizeType i = 0; i < probe_wJSON.Size(); i++){
			assert(probe_wJSON[i].IsInt());
			probe_widths.push_back(probe_wJSON[i].GetInt());
		}

		assert(val.HasMember("probe_k"));
		expressions::Expression *probe_key_expr = parseExpression(val["probe_k"]);

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
		newOp = new GpuHashJoinChained(build_e, build_widths, build_key_expr, build_op,
							probe_e, probe_widths, probe_key_expr, probe_op, hash_bits,
							dynamic_cast<GpuRawContext *>(this->ctx), maxBuildInputSize);

		build_op->setParent(newOp);
		probe_op->setParent(newOp);
	}
	else if(strcmp(opName, "join") == 0)	{
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
		}
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
//		Materializer *mat =
//				new Materializer(fieldsToMat, exprsToMat, oids, outputModes);
		Materializer* matCoarse = new Materializer(exprsToMat);

		//Put operator together
		const char *opLabel = "radixNest";
		newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels, predExpr,
				groupByExpr, nullsToZerosExpr, childOp, opLabel, *matCoarse);
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
	} else if(strcmp(opName,"block-to-tuples") == 0)	{
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
		newOp =  new BlockToTuples(childOp, ((GpuRawContext *) this->ctx), projections);
		childOp->setParent(newOp);
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

		if (gpu){
			vector<expressions::Expression *> projections;
			for (SizeType i = 0; i < val["projections"].Size(); i++){
				assert(val["projections"][i].IsObject());
				projections.push_back(this->parseExpression(val["projections"][i]));
			}

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp =  new GpuHashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr, hashAttr);
			childOp->setParent(newOp);
		} else {
			vector<RecordAttribute*> projections;
			for (SizeType i = 0; i < val["projections"].Size(); i++){
				assert(val["projections"][i].IsObject());
				projections.push_back(this->parseRecordAttr(val["projections"][i]));
			}

			assert(dynamic_cast<GpuRawContext *>(this->ctx));
			newOp =  new HashRearrange(childOp, ((GpuRawContext *) this->ctx), numOfBuckets, projections, hashExpr, hashAttr);
			childOp->setParent(newOp);
		}
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

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new MemMoveDevice(childOp, ((GpuRawContext *) this->ctx), projections);
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
		expressions::Expression * hash = NULL;
		if (val.HasMember("target")){
			assert(val["target"].IsObject());
			hash = parseExpression(val["target"]);
			numa_local = false;
		}

		if (val.HasMember("numa_local")){
			assert(numa_local);
			val["numa_local"].IsBool();
			numa_local = val["numa_local"].GetBool();
		}

		assert(dynamic_cast<GpuRawContext *>(this->ctx));
		newOp =  new Exchange(childOp, ((GpuRawContext *) this->ctx), numOfParents, projections, slack, hash, numa_local, producers);
		childOp->setParent(newOp);
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
	}
	else	{
		string err = string("Unknown Operator: ") + opName;
		LOG(ERROR) << err;
		throw runtime_error(err);
	}

	return newOp;
}

int lookupInDictionary(string s, const rapidjson::Value& val){
	assert(val.IsObject());
	assert(val.HasMember("path"));
	assert(val["path"].IsString());

	//Input Path
	const char *nameJSON = val["path"].GetString();

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

	if (!document.HasMember(s.c_str())) return -1;

	assert(document[s.c_str()].IsInt());
	return document[s.c_str()].GetInt();
}
/*
 *	enum ExpressionId	{ CONSTANT, ARGUMENT, RECORD_PROJECTION, RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE };
 *	FIXME / TODO No Merge yet!! Will be needed for parallelism!
 *	TODO Add NotExpression ?
 */

expressions::Expression* ExpressionParser::parseExpression(const rapidjson::Value& val) {

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

	if (strcmp(valExpression, "bool") == 0) {
		assert(val.HasMember("v"));
		assert(val["v"].IsBool());
		retValue = new expressions::BoolConstant(val["v"].GetBool());
	} else if (strcmp(valExpression, "int") == 0) {
		assert(val.HasMember("v"));
		assert(val["v"].IsInt());
		retValue = new expressions::IntConstant(val["v"].GetInt());
	} else if (strcmp(valExpression, "float") == 0) {
		assert(val.HasMember("v"));
		assert(val["v"].IsDouble());
		retValue = new expressions::FloatConstant(val["v"].GetDouble());
	} else if (strcmp(valExpression, "string") == 0) {
		assert(val.HasMember("v"));
		assert(val["v"].IsString());
		string *stringVal = new string(val["v"].GetString());
		retValue = new expressions::StringConstant(*stringVal);
	} else if (strcmp(valExpression, "dstring") == 0) { //FIMXE: do something better, include the dictionary
		assert(val.HasMember("v"));
		if (val["v"].IsInt()){
			retValue = new expressions::IntConstant(val["v"].GetInt());
		} else {
			assert(val["v"].IsString());
			assert(val.HasMember("dict"));

			int sVal = lookupInDictionary(val["v"].GetString(), val["dict"]);
			retValue = new expressions::IntConstant(sVal);
		}
	} else if (strcmp(valExpression, "argument") == 0) {

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

		/* e: expression over which projection is calculated */
		assert(val.HasMember(keyInnerExpr));
		assert(val[keyInnerExpr].IsObject());
		expressions::Expression *expr = parseExpression(val[keyInnerExpr]);

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
		/* exprType */
		assert(val.HasMember(keyExprType));
		assert(val[keyExprType].IsObject());
		ExpressionType *exprType = parseExpressionType(val[keyExprType]);

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
			expressions::Expression *newAttrExpr = parseExpression(attributeConstructs[i][keyAttrExpr]);

			expressions::AttributeConstruction *newAttr =
					new expressions::AttributeConstruction(newAttrName,newAttrExpr);
			newAtts->push_back(*newAttr);
		}
		retValue = new expressions::RecordConstruction(exprType,*newAtts);

	} else if (strcmp(valExpression,"if") == 0)	{
		/* exprType */
		assert(val.HasMember(keyExprType));
		assert(val[keyExprType].IsObject());
		ExpressionType *exprType = parseExpressionType(val[keyExprType]);

		/* if cond */
		assert(val.HasMember(keyCond));
		assert(val[keyCond].IsObject());
		expressions::Expression *condExpr = parseExpression(val[keyCond]);

		/* then expression */
		assert(val.HasMember(keyThen));
		assert(val[keyThen].IsObject());
		expressions::Expression *thenExpr = parseExpression(val[keyThen]);

		/* else expression */
		assert(val.HasMember(keyElse));
		assert(val[keyElse].IsObject());
		expressions::Expression *elseExpr = parseExpression(val[keyElse]);

		retValue = new expressions::IfThenElse(exprType,condExpr,thenExpr,elseExpr);
	}
	/*
	 * BINARY EXPRESSIONS
	 */
	else if (strcmp(valExpression, "eq") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::EqExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "neq") == 0) {
		/* left child */
				assert(val.HasMember(leftArg));
				assert(val[leftArg].IsObject());
				expressions::Expression *leftExpr = parseExpression(val[leftArg]);

				/* right child */
				assert(val.HasMember(rightArg));
				assert(val[rightArg].IsObject());
				expressions::Expression *rightExpr = parseExpression(val[rightArg]);

				retValue = new expressions::NeExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "lt") == 0) {
		/* left child */
				assert(val.HasMember(leftArg));
				assert(val[leftArg].IsObject());
				expressions::Expression *leftExpr = parseExpression(val[leftArg]);

				/* right child */
				assert(val.HasMember(rightArg));
				assert(val[rightArg].IsObject());
				expressions::Expression *rightExpr = parseExpression(val[rightArg]);

				retValue = new expressions::LtExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "le") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::LeExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "gt") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::GtExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "ge") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::GeExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "and") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::AndExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "or") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		retValue = new expressions::OrExpression(new BoolType(),leftExpr,rightExpr);
	} else if (strcmp(valExpression, "add") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::AddExpression(exprType,leftExpr,rightExpr);
	} else if (strcmp(valExpression, "sub") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::SubExpression(exprType,leftExpr,rightExpr);
	} else if (strcmp(valExpression, "multiply") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::MultExpression(exprType,leftExpr,rightExpr);
	} else if (strcmp(valExpression, "div") == 0) {
		/* left child */
		assert(val.HasMember(leftArg));
		assert(val[leftArg].IsObject());
		expressions::Expression *leftExpr = parseExpression(val[leftArg]);

		/* right child */
		assert(val.HasMember(rightArg));
		assert(val[rightArg].IsObject());
		expressions::Expression *rightExpr = parseExpression(val[rightArg]);

		ExpressionType *exprType = const_cast<ExpressionType*>(leftExpr->getExpressionType());
		retValue = new expressions::DivExpression(exprType,leftExpr,rightExpr);
	} else if (strcmp(valExpression, "merge") == 0) {
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
		RecordAttribute * reg_as = parseRecordAttr(val["register_as"]);

		InputInfo * datasetInfo = (this->catalogParser).getInputInfo(reg_as->getRelationName());
		RecordType * rec = new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType *>(datasetInfo->exprType)->getNestedType())};

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
	} else if (strcmp(valExprType, "string") == 0) {
		return new StringType();
	} else if (strcmp(valExprType, "dstring") == 0) {
		return new IntType();
	} else if (strcmp(valExprType, "set") == 0) {
		assert(val.HasMember("inner"));
		assert(val["inner"].IsObject());
		ExpressionType *innerType = parseExpressionType(val["inner"]);
		return new SetType(*innerType);
	} else if (strcmp(valExprType, "bag") == 0) {
		assert(val.HasMember("inner"));
		assert(val["inner"].IsObject());
		ExpressionType *innerType = parseExpressionType(val["inner"]);
		cout << "BAG" << endl;
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
		} else {
			assert(val.HasMember("relName"));
			assert(val["relName"].IsString());

			return getRecordType(val["relName"].GetString());
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

RecordAttribute* ExpressionParser::parseRecordAttr(const rapidjson::Value& val) {

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
		if (!attr) std::cout << relName << "." << attrName << std::endl;
		assert(attr && "Attribute not found");
		attrNo = attr->getAttrNo();
	}

	const ExpressionType* recArgType;
	if (val.HasMember(keyRecAttrType)){
		assert(val[keyRecAttrType].IsObject());
		recArgType = parseExpressionType(val[keyRecAttrType]);
	} else {
		assert(attr && "Attribute not found");
		recArgType = attr->getOriginalType();
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
	InputInfo *datasetInfo = (this->catalogParser).getInputInfo(datasetName);
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

/**
 * {"datasetname": {"path": "foo", "type": { ... } }
 */
CatalogParser::CatalogParser(const char *catalogPath): exprParser(*this) {
	//Input Path
	const char *nameJSON = catalogPath;

	//key aliases
	const char *keyInputPath = "path";
	const char *keyExprType =  "type";

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
		const char *err = "[CatalogParser: ] Error parsing physical plan";
		LOG(ERROR)<< err;
		throw runtime_error(err);
	}

	//Start plan traversal.
	printf("\nParsing catalog information:\n");
	assert(document.IsObject());

	for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin();
			itr != document.MemberEnd(); ++itr) {
		printf("Key of member is %s\n", itr->name.GetString());

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
//		(this->inputs)[itr->name.GetString()] = info;
		(this->inputs)[info->path] = info;
	}
}
