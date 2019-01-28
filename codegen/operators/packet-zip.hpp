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

#ifndef PACKET_ZIP_HPP_
#define PACKET_ZIP_HPP_

#include <unordered_map>
#include "expressions/expressions.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

struct ZipParam {
  int heads_id;
  int sizes_id;
  int oids_id;
  int blocks_id;
  int chains_id;
  int offset_id;
};

struct ZipState {
  int64_t *cnts[128];
  int64_t *oids[128];
  void **blocks[128];
  int32_t *blocks_chain[128];
  int32_t *blocks_head[128];
};

class ZipCollect : public BinaryRawOperator {
 public:
  ZipCollect(RecordAttribute *ptrAttr, RecordAttribute *splitter,
             RecordAttribute *targetAttr, RecordAttribute *inputLeft,
             RecordAttribute *inputRight, RawOperator *const leftChild,
             RawOperator *const rightChild, GpuRawContext *const context,
             int numOfBuckets, RecordAttribute *hash_key_left,
             const vector<expression_t> &wantedFieldsLeft,
             RecordAttribute *hash_key_right,
             const vector<expression_t> &wantedFieldsRight, string opLabel);

  virtual ~ZipCollect() { LOG(INFO) << "Collapsing PacketZip operator"; }

  virtual void produce();
  virtual void consume(RawContext *const context,
                       const OperatorState &childState);

  void generate_cache_left(RawContext *const context,
                           const OperatorState &childState);
  void generate_cache_right(RawContext *const context,
                            const OperatorState &childState);

  void open_cache_left(RawPipeline *pip);
  void close_cache_left(RawPipeline *pip);

  void open_cache_right(RawPipeline *pip);
  void close_cache_right(RawPipeline *pip);

  void open_pipe(RawPipeline *pip);
  void close_pipe(RawPipeline *pip);
  void ctrl(RawPipeline *pip);

  ZipState &getStateLeft() { return state_left; }
  ZipState &getStateRight() { return state_right; }

  virtual bool isFiltering() const { return false; }

 private:
  void cacheFormatLeft();
  void cacheFormatRight();
  void pipeFormat();

  void generate_send();

  int *partition_ptr[128];

  GpuRawContext *context;
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

  int partition_id;

  ZipParam cache_left_p;
  ZipParam cache_right_p;
  ZipParam pipe_left_p;
  ZipParam pipe_right_p;
};

class ZipInitiate : public UnaryRawOperator {
 public:
  ZipInitiate(RecordAttribute *ptrAttr, RecordAttribute *splitter,
              RecordAttribute *targetAttr, RawOperator *const child,
              GpuRawContext *const context, int numOfBuckets, ZipState &state1,
              ZipState &state2, string opLabel);

  virtual ~ZipInitiate() {}

  virtual void produce();

  virtual void consume(RawContext *const context,
                       const OperatorState &childState);

  virtual bool isFiltering() const { return false; }

  RawPipelineGen **pipeSocket() { return &join_pip; }

  void open_fwd(RawPipeline *pip);
  void close_fwd(RawPipeline *pip);
  void open_cache(RawPipeline *pip);
  void close_cache(RawPipeline *pip);
  void ctrl(RawPipeline *pip);

 private:
  void generate_send();

  GpuRawContext *context;
  string opLabel;
  RecordAttribute *targetAttr;
  RecordAttribute *ptrAttr;
  RecordAttribute *splitter;
  int numOfBuckets;

  int *partition_ptr[128];
  int *partitions[128];

  int partition_alloc_cache;
  int partition_cnt_cache;

  int right_blocks_id;
  int left_blocks_id;

  int partition_fwd;

  int calls;

  ZipState &state1;
  ZipState &state2;

  RawPipelineGen *join_pip;
  std::vector<RawPipelineGen *> launch;
};

class ZipForward : public UnaryRawOperator {
 public:
  ZipForward(RecordAttribute *splitter, RecordAttribute *targetAttr,
             RecordAttribute *inputAttr, RawOperator *const child,
             GpuRawContext *const context, int numOfBuckets,
             const vector<expression_t> &wantedFields, string opLabel,
             ZipState &state);

  virtual ~ZipForward() {}

  virtual void produce();
  virtual void consume(RawContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  void open(RawPipeline *pip);
  void close(RawPipeline *pip);

 private:
  void cacheFormat();

  GpuRawContext *context;
  string opLabel;
  vector<expression_t> wantedFields;
  RecordAttribute *inputAttr;
  RecordAttribute *targetAttr;
  RecordAttribute *splitter;
  int numOfBuckets;

  int *partition_ptr[128];
  int *partitions[128];

  ZipState &state;

  int partition_alloc;
  int partition_cnt;

  ZipParam p;
};

#endif