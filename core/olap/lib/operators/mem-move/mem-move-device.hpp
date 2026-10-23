/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#ifndef MEM_MOVE_DEVICE_HPP_
#define MEM_MOVE_DEVICE_HPP_

#include <future>
#include <olap/util/parallel-context.hpp>
#include <platform/memory/managed-pointer.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/async_containers.hpp>
#include <storage/io_uring.hpp>
#include <thread>

#include "compression.hpp"
#include "lib/operators/operators.hpp"
#include "olap/plugins/binary-block-nvme-plugin.hpp"

struct buff_pair {
  proteus::managed_ptr new_buff;
  proteus::managed_ptr old_buff;

  static buff_pair not_moved(proteus::managed_ptr buff);
  [[nodiscard]] bool moved() const;
};

class MemMoveDevice : public experimental::UnaryOperator {
 public:
  struct alignas(64) workunit {
    void *data;
    cudaEvent_t event;
    std::atomic<int> complete;  // 0 for complete, positive number for remaining
    ssize_t *bytes_read;        // for cufile async api only
    cudaStream_t cufile_strm;   // stream for cufile/nvcomp, memcpyasync still
                                // uses mmc->strm
    std::vector<GpuDecompressor> *decompressors;
    uint8_t index_in_wu;           // Index within bytes_read
    [[maybe_unused]] bool unused;  // FIXME: remove
    // GPU scratch buffers used only by the NVMe->GPU compressed batch path.
    // Layout: [gpu_idx][field_idx] -> device pointer where compressed bytes
    // are read (cuFile) before batch_decompress_block_gpu() writes into a
    // BlockManager output block.
    // Vectors are allocated per workunit to avoid aliasing between concurrent
    // in-flight workunits. For NONE/deferred paths, inner vectors may stay
    // empty.
    std::vector<std::vector<void*>> *compressed_buffers;
  };

  /**
   * Between opening and closing of a pipeline MemMoveDevice itself is largely
   * stateless. All of the state is instead stored in MemMoveConf.
   */
  class MemMoveConf {
   public:
    // Idle workunits ready to be aquired
    AsyncQueueSPSC<workunit *>
        idle;  //_lockfree is slower and seems to have a bug
    /**
     * tran holds workunits that have been propogated (pushed) while they wait
     * to be reaped with getPropagated (pulled)
     */
    AsyncQueueSPSC<workunit *> tran;
    /**
     * Handle for the thread catching propagated workunits and calling `pull`
     */
    std::future<void> worker;
    cudaStream_t strm = nullptr;
    cudaEvent_t *lastEvent = nullptr;
    // The number of workunits in circulation
    size_t slack = 0;
    size_t cnt = 0;
    // The pinned data backing the workunits
    void *data_buffs = nullptr;
    std::unique_ptr<proteus::storage::IoUringThreadUnsafe> io_uring{};
    NvmePlugin *nvme_plugin = nullptr;
    std::vector<bool> do_transfer{};
    bool enable_deferred_rfor_decode{false};

   public:
    virtual ~MemMoveConf() = default;

    virtual buff_pair push(proteus::managed_ptr src, size_t bytes,
                           int target_device, uint64_t srcServer, workunit *wu);
    virtual proteus::managed_ptr force_push(const proteus::managed_ptr &src,
                                            size_t bytes, int target_device,
                                            uint64_t srcServer,
                                            cudaStream_t movestrm);
    proteus::managed_ptr force_push_from_nvme(const proteus::managed_ptr &src,
                                              int target_device,
                                              cudaStream_t movestrm,
                                              workunit *wu);
    virtual proteus::managed_ptr pull(proteus::managed_ptr buff) {
      return buff;
    }
    std::vector<buff_pair> batch_push_nvme_to_gpu(
        const std::vector<proteus::managed_ptr> &src, size_t bytes,
        int target_device, uint64_t srcServer, workunit *wu);

    /**
     * acquire an idle workunit
     * @return a workunit
     */
    virtual MemMoveDevice::workunit *acquire();
    /**
     * Propagate a workunit onto the tran queue so that he can be caught with
     * `getPropagated` The work unit must be previously obtained with `acquire`
     */
    virtual void propagate(MemMoveDevice::workunit *buff, bool is_noop);

    /**
     * This will block until either a workunit has been obtained, or the
     * operator is terminating.
     * @param ret pointer to a workunit pointer which will be set to the value
     * of a propagated workunit.
     * @return true if a workunit has been obtained, and false if there are no
     * more workunits to obtain and the operator is terminating.
     */
    virtual bool getPropagated(MemMoveDevice::workunit **ret);

    /**
     * Release a workunit into the idle pool
     * @param buff a workunit
     */
    virtual void release(MemMoveDevice::workunit *buff);
  };

  MemMoveDevice(Operator *const child,
                const std::vector<RecordAttribute *> &wantedFields, size_t slack,
                bool to_cpu, std::vector<bool> do_transfer)
      : UnaryOperator(child),
        wantedFields(wantedFields),
        slack(slack),
        to_cpu(to_cpu),
        do_transfer(std::move(do_transfer)) {}

  MemMoveDevice(Operator *const child,
                const std::vector<RecordAttribute *> &wantedFields, size_t slack,
                bool to_cpu)
      : MemMoveDevice(child, wantedFields, slack, to_cpu,
                      //                      {true, true, false, false}
                      std::vector<bool>(wantedFields.size(), true)) {}

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override { return wantedFields; }

 protected:
  [[nodiscard]] virtual MemMoveConf *createMoveConf() const;
  virtual void destroyMoveConf(MemMoveConf *mmc) const;

  [[nodiscard]] virtual ProteusValueMemory getServerId(
      OlapParallelContext *context, const OperatorState &childState) const;

  virtual void genReleaseOldBuffer(OlapParallelContext *context,
                                   llvm::Value *pValue) const;

  /* Code for batch decompression */
  const uint32_t batch_size = 1;
  StateVar batch_cnt_ptr; // Change this to current batch
  StateVar tran_empty_ptr;
  StateVar mv_src_ptrs_arr_ptr;
  StateVar mv_sizes_bytes_arr_ptr;
  StateVar num_tuples_arr_ptr;
  StateVar oid_arr_ptr;
  /* End of code for batch decompression */

  const std::vector<RecordAttribute *> wantedFields;
  StateVar device_id_var;
  StateVar memmvconf_var;

  PipelineGen *catch_pip;
  llvm::Type *data_type;

  size_t slack;
  bool to_cpu;

  std::vector<bool> do_transfer;

  void catcher(MemMoveConf *conf, int group_id, exec_location target_dev,
               const void *session);

 protected:
  [[nodiscard]] int getTargetDevice() const;
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);
  // profiling domain
  struct memmove_domain {
    static constexpr char const *name{"memmove"};
  };
};

extern "C" {
MemMoveDevice::workunit *acquireWorkUnit(
    MemMoveDevice::MemMoveConf *mmc) noexcept;
}

#endif /* MEM_MOVE_DEVICE_HPP_ */
