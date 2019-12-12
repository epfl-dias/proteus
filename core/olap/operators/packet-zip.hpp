/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#ifndef PACKET_ZIP_HPP_
#define PACKET_ZIP_HPP_

#include <unordered_map>

#include "expressions/expressions.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/operators.hpp"
#include "util/parallel-context.hpp"

struct ZipParam {
  StateVar heads_id;
  StateVar sizes_id;
  StateVar oids_id;
  StateVar blocks_id;
  StateVar chains_id;
  StateVar offset_id;
};

struct ZipState {
  int64_t *cnts[128];
  int64_t *oids[128];
  void **blocks[128];
  int32_t *blocks_chain[128];
  int32_t *blocks_head[128];
};

class ZipCollect : public BinaryOperator {
 public:
  ZipCollect(RecordAttribute *ptrAttr, RecordAttribute *splitter,
             RecordAttribute *targetAttr, RecordAttribute *inputLeft,
             RecordAttribute *inputRight, Operator *const leftChild,
             Operator *const rightChild, ParallelContext *const context,
             int numOfBuckets, RecordAttribute *hash_key_left,
             const vector<expression_t> &wantedFieldsLeft,
             RecordAttribute *hash_key_right,
             const vector<expression_t> &wantedFieldsRight, string opLabel);

  virtual ~ZipCollect() { LOG(INFO) << "Collapsing PacketZip operator"; }

  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);

  void generate_cache_left(Context *const context,
                           const OperatorState &childState);
  void generate_cache_right(Context *const context,
                            const OperatorState &childState);

  void open_cache_left(Pipeline *pip);
  void close_cache_left(Pipeline *pip);

  void open_cache_right(Pipeline *pip);
  void close_cache_right(Pipeline *pip);

  void open_pipe(Pipeline *pip);
  void close_pipe(Pipeline *pip);
  void ctrl(Pipeline *pip);

  ZipState &getStateLeft() { return state_left; }
  ZipState &getStateRight() { return state_right; }

  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void cacheFormatLeft();
  void cacheFormatRight();
  void pipeFormat();

  void generate_send();

  int *partition_ptr[128];

  ParallelContext *context;
  string opLabel;
  vector<expression_t> wantedFieldsLeft;
  vector<expression_t> wantedFieldsRight;
  RecordAttribute *splitter;
  RecordAttribute *hash_key_left;
  RecordAttribute *hash_key_right;
  RecordAttribute *inputLeft;
  RecordAttribute *inputRight;
  RecordAttribute *targetAttr;
  RecordAttribute *ptrAttr;
  int numOfBuckets;

  ZipState state_right;

  ZipState state_left;

  int *offset_left[128];
  int *offset_right[128];

  StateVar partition_id;

  ZipParam cache_left_p;
  ZipParam cache_right_p;
  ZipParam pipe_left_p;
  ZipParam pipe_right_p;
};

class ZipInitiate : public UnaryOperator {
 public:
  ZipInitiate(RecordAttribute *ptrAttr, RecordAttribute *splitter,
              RecordAttribute *targetAttr, Operator *const child,
              ParallelContext *const context, int numOfBuckets,
              ZipState &state1, ZipState &state2, string opLabel);

  virtual ~ZipInitiate() {}

  virtual void produce_(ParallelContext *context);

  virtual void consume(Context *const context, const OperatorState &childState);

  virtual bool isFiltering() const { return false; }

  PipelineGen **pipeSocket() { return &join_pip; }

  void open_fwd(Pipeline *pip);
  void close_fwd(Pipeline *pip);
  void open_cache(Pipeline *pip);
  void close_cache(Pipeline *pip);
  void ctrl(Pipeline *pip);

  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void generate_send();

  ParallelContext *context;
  string opLabel;
  RecordAttribute *targetAttr;
  RecordAttribute *ptrAttr;
  RecordAttribute *splitter;
  int numOfBuckets;

  int *partition_ptr[128];
  int *partitions[128];

  StateVar partition_alloc_cache;
  StateVar partition_cnt_cache;

  StateVar right_blocks_id;
  StateVar left_blocks_id;

  StateVar partition_fwd;

  int calls;

  ZipState &state1;
  ZipState &state2;

  PipelineGen *join_pip;
  std::vector<PipelineGen *> launch;
};

class ZipForward : public UnaryOperator {
 public:
  ZipForward(RecordAttribute *targetAttr, Operator *const child,
             ParallelContext *const context,
             const vector<expression_t> &wantedFields, string opLabel,
             ZipState &state);

  virtual ~ZipForward() {}

  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void cacheFormat();

  ParallelContext *context;
  string opLabel;
  vector<expression_t> wantedFields;
  RecordAttribute *targetAttr;

  ZipState &state;

  ZipParam p;
};

#endif
