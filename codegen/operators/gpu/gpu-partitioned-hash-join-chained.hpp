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

#include "codegen/util/parallel-context.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/operators.hpp"

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
  HashPartitioner(RecordAttribute *targetAttr,
                  const std::vector<GpuMatExpr> &parts_mat_exprs,
                  const std::vector<size_t> &parts_packet_widths,
                  expression_t parts_keyexpr, Operator *const parts_child,
                  ParallelContext *context, size_t maxInputSize, int log_parts,
                  std::string opLabel);

  virtual ~HashPartitioner() {}

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);

  virtual bool isFiltering() const { return true; }

  PartitionState &getState() { return state; }

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  llvm::StructType *getPayloadType() { return payloadType; }

 private:
  void matFormat();

  size_t maxInputSize;

  int log_parts;
  int log_parts1;
  int log_parts2;

  OutputPlugin *pg_out;

  llvm::StructType *payloadType;

  int32_t *cnts_ptr[128];

  PartitionState state;

  std::vector<GpuMatExpr> parts_mat_exprs;
  std::vector<size_t> parts_packet_widths;
  expression_t parts_keyexpr;

  std::vector<int> param_pipe_ids;

  int cnt_pipe;

  ParallelContext *context;

  RecordAttribute *targetAttr;

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

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);

  void open(Pipeline *pip);
  void close(Pipeline *pip);
  void allocate(Pipeline *pip);

  virtual bool isFiltering() const { return true; }

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

  int head_id;

  std::vector<int> build_param_pipe_ids;
  std::vector<int> probe_param_pipe_ids;

  std::vector<int> build_param_join_ids;
  std::vector<int> probe_param_join_ids;

  int cnt_left_pipe;
  int cnt_right_pipe;

  int cnt_left_join;
  int cnt_right_join;

  int chains_left_join;
  int chains_right_join;

  int keys_partitioned_probe_id;
  int idxs_partitioned_probe_id;

  int keys_partitioned_build_id;
  int idxs_partitioned_build_id;

  int keys_cache_id;
  int next_cache_id;
  int idxs_cache_id;

  int32_t *buffer[128];
  int buffer_id;

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

  int bucket_info_id;
  int buckets_used_id;

  // OperatorState childState;
  // OperatorState leftState;

  // GpuExprMaterializer *   build_mat  ;
  // GpuExprMaterializer *   probe_mat  ;
  ParallelContext *context;
};

#endif /* GPU_HASH_JOIN_CHAINED_HPP_ */
