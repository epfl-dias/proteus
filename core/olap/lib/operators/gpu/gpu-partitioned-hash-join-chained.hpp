/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef GPU_PHASH_JOIN_CHAINED_HPP_
#define GPU_PHASH_JOIN_CHAINED_HPP_

#include <optional>
#include <unordered_map>

#include "common/gpu/gpu-common.hpp"
#include "lib/operators/operators.hpp"
#include "lib/util/jit/pipeline.hpp"
#include "olap/operators/gpu/gpu-materializer-expr.hpp"
#include "olap/util/parallel-context.hpp"

struct PartitionMetadata {
  int32_t *keys;
  int32_t *payload;
  uint32_t *chains;
  uint32_t *bucket_info;
  int32_t *out_cnts;
  uint32_t *buckets_used;
  uint64_t *heads;
};

struct PartitionState {
  int *atom_cnt[128];
  char *allocas[128];
  std::vector<void *> cols[128];
  PartitionMetadata meta[128];
};

class HashPartitioner : public UnaryOperator {
 public:
  HashPartitioner(const std::vector<GpuMatExpr> &parts_mat_exprs,
                  const std::vector<size_t> &parts_packet_widths,
                  expression_t parts_keyexpr, Operator *const parts_child,
                  ParallelContext *context, size_t maxInputSize, int log_parts,
                  std::string opLabel);

  virtual ~HashPartitioner() {}

  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);

  virtual bool isFiltering() const { return true; }

  PartitionState &getState() { return state; }

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  llvm::StructType *getPayloadType() { return payloadType; }

  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void matFormat();

  size_t maxInputSize;

  int log_parts1;
  int log_parts2;

  llvm::StructType *payloadType;

  int32_t *cnts_ptr[128];

  PartitionState state;

  std::vector<GpuMatExpr> parts_mat_exprs;
  std::vector<size_t> parts_packet_widths;
  expression_t parts_keyexpr;

  std::vector<StateVar> param_pipe_ids;

  StateVar cnt_pipe;

  ParallelContext *context;

  std::string opLabel;
};

class GpuPartitionedHashJoinChained : public BinaryOperator {
 public:
  GpuPartitionedHashJoinChained(
      const std::vector<GpuMatExpr> &build_mat_exprs,
      const std::vector<size_t> &build_packet_widths,
      expression_t build_keyexpr,
      std::optional<expression_t> build_minor_keyexpr,
      HashPartitioner *const build_child,

      const std::vector<GpuMatExpr> &probe_mat_exprs,
      const std::vector<size_t> &probe_mat_packet_widths,
      expression_t probe_keyexpr,
      std::optional<expression_t> probe_minor_keyexpr,
      HashPartitioner *const probe_child,

      PartitionState &state_left, PartitionState &state_right,

      size_t maxBuildInputSize, size_t maxProbeInputSize,

      int log_parts, ParallelContext *context,
      std::string opLabel = "hj_chained", PipelineGen **caller = nullptr,
      Operator *const unionop = nullptr);
  virtual ~GpuPartitionedHashJoinChained() {
    LOG(INFO) << "Collapsing GpuOptJoin operator";
  }

  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);

  void open(Pipeline *pip);
  void close(Pipeline *pip);
  void allocate(Pipeline *pip);

  virtual bool isFiltering() const { return true; }

  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void generate_materialize_left(Context *const context,
                                 const OperatorState &childState);
  void generate_materialize_right(Context *const context,
                                  const OperatorState &childState);

  void generate_mock1(Context *const context);
  void generate_joinloop(Context *const context);
  void generate_build(Context *const context,
                      std::map<std::string, llvm::Value *> &kernelBindings);
  void generate_probe(Context *const context,
                      std::map<std::string, llvm::Value *> &kernelBindings);
  void generate(Context *const context);
  void buildHashTableFormat();
  void probeHashTableFormat();
  void matLeftFormat();
  void matRightFormat();

  llvm::Value *hash(llvm::Value *key);

  PipelineGen **caller;
  std::string opLabel;

  llvm::StructType *payloadType_left;
  llvm::StructType *payloadType_right;

  std::vector<GpuMatExpr> build_mat_exprs;
  std::vector<GpuMatExpr> probe_mat_exprs;
  std::vector<size_t> build_packet_widths;

  expression_t build_keyexpr;
  expression_t probe_keyexpr;

  std::optional<expression_t> build_minor_keyexpr;
  std::optional<expression_t> probe_minor_keyexpr;

  std::vector<size_t> packet_widths;

  StateVar head_id;

  std::vector<StateVar> build_param_pipe_ids;
  std::vector<StateVar> probe_param_pipe_ids;

  std::vector<StateVar> build_param_join_ids;
  std::vector<StateVar> probe_param_join_ids;

  StateVar cnt_left_pipe;
  StateVar cnt_right_pipe;

  StateVar cnt_left_join;
  StateVar cnt_right_join;

  StateVar chains_left_join;
  StateVar chains_right_join;

  StateVar keys_partitioned_probe_id;
  StateVar idxs_partitioned_probe_id;

  StateVar keys_partitioned_build_id;
  StateVar idxs_partitioned_build_id;

  StateVar keys_cache_id;
  StateVar next_cache_id;
  StateVar idxs_cache_id;

  int32_t *buffer[128];
  StateVar buffer_id;

  llvm::BasicBlock *boot;
  llvm::BasicBlock *start;

  int log_parts;
  int log_parts1;
  int log_parts2;

  PartitionState &state_left;
  PartitionState &state_right;

  Operator *unionop;

  int hash_bits;
  size_t maxBuildInputSize;
  size_t maxProbeInputSize;

  cudaEvent_t jstart[128];
  cudaEvent_t jstop[128];

  StateVar bucket_info_id;
  StateVar buckets_used_id;

  // OperatorState childState;
  // OperatorState leftState;

  // GpuExprMaterializer *   build_mat  ;
  // GpuExprMaterializer *   probe_mat  ;
  ParallelContext *context;
};

#endif /* GPU_HASH_JOIN_CHAINED_HPP_ */
