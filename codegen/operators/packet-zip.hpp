#ifndef PACKET_ZIP_HPP_
#define PACKET_ZIP_HPP_

#include "util/gpu/gpu-raw-context.hpp"
#include "operators/operators.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "expressions/expressions.hpp"
#include <unordered_map> 


struct ZipParam {
	int heads_id; 
	int sizes_id;
	int oids_id;
	int blocks_id;
	int chains_id;
	int offset_id;
};

struct ZipState {
	int64_t* cnts[128];
	int64_t* oids[128];
	void** blocks[128];
	int32_t* blocks_chain[128];
	int32_t* blocks_head[128];
};


class ZipCollect : public BinaryRawOperator {
public:
	ZipCollect (		RecordAttribute*                            ptrAttr,
						RecordAttribute*                            splitter,
						RecordAttribute*                            targetAttr,
						RecordAttribute*                            inputLeft,
						RecordAttribute*                            inputRight,
						RawOperator * const             			leftChild,
						RawOperator * const             			rightChild,
                    	GpuRawContext * const           			context,
                    	int                             			numOfBuckets,
                    	RecordAttribute*                            hash_key_left,
                    	const vector<expressions::Expression*>&	wantedFieldsLeft,
                    	RecordAttribute*                            hash_key_right,
                    	const vector<expressions::Expression*>&	wantedFieldsRight,
                    	string                          			opLabel);

	virtual ~ZipCollect() { LOG(INFO)<< "Collapsing PacketZip operator";}

	virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);

    void generate_cache_left(RawContext* const context, const OperatorState& childState);
    void generate_cache_right(RawContext* const context, const OperatorState& childState);

    void open_cache_left (RawPipeline* pip);
    void close_cache_left(RawPipeline* pip);

    void open_cache_right (RawPipeline* pip);
    void close_cache_right(RawPipeline* pip);

    void open_pipe (RawPipeline* pip);
    void close_pipe (RawPipeline* pip);
    void ctrl (RawPipeline* pip);

    ZipState& getStateLeft ()  { return state_left; }
    ZipState& getStateRight () { return state_right; }

    virtual bool isFiltering() const {return false;}
private:
	void cacheFormatLeft();
	void cacheFormatRight();
	void pipeFormat();


	void generate_send();
	
	int* partition_ptr[128];

	GpuRawContext* context;
	string opLabel;
	vector<expressions::Expression*>	wantedFieldsLeft;
    vector<expressions::Expression*>	wantedFieldsRight;
    RecordAttribute*            splitter;
    RecordAttribute*            hash_key_left;
    RecordAttribute*            hash_key_right;
    RecordAttribute*            inputLeft;
    RecordAttribute*            inputRight;
    RecordAttribute*            targetAttr;
    RecordAttribute*            ptrAttr;
    int numOfBuckets;

    ZipState state_right;

	ZipState state_left;

	int* offset_left[128];
	int* offset_right[128];

	int partition_id;


	ZipParam cache_left_p;
	ZipParam cache_right_p;
	ZipParam pipe_left_p;
	ZipParam pipe_right_p;
};

class ZipInitiate : public UnaryRawOperator {
public:
	ZipInitiate (
						RecordAttribute *                           ptrAttr,
                        RecordAttribute *                           splitter,
						RecordAttribute *							targetAttr,
						RawOperator * const                         child,
						GpuRawContext * const                       context,
						int 										numOfBuckets,
						ZipState& 									state1,
						ZipState& 									state2,
						string										opLabel
				);

	virtual ~ZipInitiate () {}

	virtual void produce ();

	virtual void consume(RawContext* const context, const OperatorState& childState);

	virtual bool isFiltering() const {return false;}

	RawPipelineGen** pipeSocket () { return &join_pip; }

	void open_fwd (RawPipeline* pip);
	void close_fwd(RawPipeline* pip);
	void open_cache (RawPipeline* pip);
	void close_cache(RawPipeline* pip);
	void ctrl (RawPipeline* pip);

private:

	void generate_send();

	GpuRawContext* context;
	string opLabel;
	RecordAttribute*            targetAttr;
	RecordAttribute*            ptrAttr;
	RecordAttribute*            splitter;
	int numOfBuckets;

	int* partition_ptr[128];
	int* partitions[128];

	int partition_alloc_cache;
	int partition_cnt_cache;

	int right_blocks_id;
	int left_blocks_id;

	int partition_fwd;

	int calls;

	ZipState& state1;
	ZipState& state2;

	RawPipelineGen * join_pip;
	std::vector<RawPipelineGen *> launch;
};

class ZipForward : public UnaryRawOperator {
public:
	ZipForward (		RecordAttribute*                            splitter,
						RecordAttribute*                            targetAttr,
						RecordAttribute*                            inputAttr,
						RawOperator * const	             			child,
                    	GpuRawContext * const           			context,
                    	int                             			numOfBuckets,
                    	const vector<expressions::Expression*>&		wantedFields,
                    	string                          			opLabel,
                    	ZipState&                                   state);

	virtual ~ZipForward () {}

	virtual void produce ();
	virtual void consume(RawContext* const context, const OperatorState& childState);
	virtual bool isFiltering() const {return false;}

	void open (RawPipeline* pip);
    void close (RawPipeline* pip);

    

private:
	void cacheFormat();

	GpuRawContext* context;
	string opLabel;
	vector<expressions::Expression*>	wantedFields;
    RecordAttribute*            inputAttr;
    RecordAttribute*            targetAttr;
    RecordAttribute*                            splitter;
    int numOfBuckets;

	int* partition_ptr[128];
	int* partitions[128];

	ZipState& state;

	int partition_alloc;
	int partition_cnt;

	ZipParam p;
};


#endif