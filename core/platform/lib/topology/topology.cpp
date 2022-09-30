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

#include <numa.h>
#include <numaif.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <platform/common/error-handling.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/logging.hpp>
#include <platform/util/topology_parser.hpp>
#include <regex>
#include <stdexcept>
#include <vector>

#include "cuda_profiler_api.h"
#include "cuda_runtime_api.h"
#include "nvToolsExt.h"
#include "nvml.h"

// template<typename T>
const topology::cpunumanode *topology::getCpuNumaNodeAddressed(
    const void *m) const {
  int numa_id = -1;
  auto x = get_mempolicy(&numa_id, nullptr, 0, const_cast<void *>(m),
                         MPOL_F_NODE | MPOL_F_ADDR);
  if (x) return nullptr;
  assert(numa_id >= 0);
  return (cpu_info.data() + cpunuma_index[numa_id]);
}

// the @ in the link really messes with clang documentation parsing
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-documentation-unknown-command"
/**
 * There are two thing that /sys/block/nvme2n1 may link to.
 * Either a virtual NVMe device, or a physical NVMe Device. Virtual devices
 * cover NVMe over fabric, which we do not yet support. We just want the
 * physical device. The core issue is that the pci devices and
 * the virtual devices have different folder structures with the same
 * information.
 * @see
 * https://lore.kernel.org/all/3c725e5deaabaaf145f48f2f6fcfdae9f6d41e2e.camel@suse.de/t/#r543f7392cfa1ba8adfe5447735b8da6170022295
 * @param nvmeBlockDevice e.g /sys/block/nvme2n1
 * @return a path which refers to the physical pci device directory of the nvme
 * drive, e.g a path which symlinks to:
 * /sys/devices/pci0000:17/0000:17:00.0/0000:18:00.0/nvme/nvme2/device
 */
#pragma clang diagnostic pop
std::filesystem::path nvmeNameSpaceToSysfsPciSysfsPath(
    const std::string &nvmeBlockDevice) {
  std::regex thisDeviceRegex{"(nvme[0-9]+).*"};
  std::smatch extract_device_regex;
  assert(std::regex_search(nvmeBlockDevice, extract_device_regex,
                           thisDeviceRegex));
  // should get us just nvme1
  auto thisDeviceOnly = extract_device_regex[1].str();

  std::vector<std::filesystem::path> possibleVirtualDevicePaths{};
  std::vector<std::filesystem::path> possiblePhysicalDevicePaths{};
  // If we only have nvme block devices in the form `nvme1n1` then those are
  // physical devices e.g they symlink to
  // `../devices/pci0000:80/0000:80:03.2/0000:84:00.0/nvme/nvme1/nvme1n1` if we
  // have both `nvme1n1` and `nvme1c1n1` then nvme1c1n1 is the symlink to the
  // pcie device

  std::regex possibleVirtualNvmeRegex{R"(\/sys\/block\/nvme[0-9]+n[0-9]+$)"};
  std::regex possiblePhysicalNvmeRegex{
      R"(\/sys\/block\/nvme[0-9]+c[0-9]+n[0-9]+$)"};
  for (const auto &blockDevice :
       std::filesystem::directory_iterator("/sys/block")) {
    const auto devString = blockDevice.path().string();
    std::smatch nvme_maybe_virtual_match;
    std::smatch nvme_maybe_physical_match;
    std::smatch this_device_match;

    std::regex_search(devString, this_device_match, thisDeviceRegex);
    if (this_device_match[1].str() == thisDeviceOnly) {
      if (std::regex_search(devString, nvme_maybe_virtual_match,
                            possibleVirtualNvmeRegex)) {
        possibleVirtualDevicePaths.emplace_back(blockDevice.path().string());
      }

      if (std::regex_search(devString, nvme_maybe_physical_match,
                            possiblePhysicalNvmeRegex)) {
        possiblePhysicalDevicePaths.emplace_back(blockDevice.path().string());
      }
    }
  }

  if (possiblePhysicalDevicePaths.size() == 1) {
    return possiblePhysicalDevicePaths.at(0) / "device";
  } else {
    assert(possibleVirtualDevicePaths.size() == 1 &&
           possiblePhysicalDevicePaths.empty());
    return possibleVirtualDevicePaths.at(0) / "device";
  }
}

class nvme_info_discovery_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

uint32_t topology::pcieAddressToNumaNodeId(const std::string &address) {
  std::string filename = "/sys/bus/pci/devices/" + address + "/numa_node";
  std::ifstream s(filename);
  if (!s.is_open()) {
    throw std::runtime_error("Failed to open: " + filename +
                             ". Is this a valid pcie address?");
  }
  std::string numa_node;
  s >> numa_node;
  return std::stoul(numa_node);
}

uint32_t nvmeDevPathToNumaNodeId(const std::string &devPath) {
  std::filesystem::path filename =
      nvmeNameSpaceToSysfsPciSysfsPath(devPath) / "device" / "numa_node";

  std::ifstream s(filename);
  if (!s.is_open()) {
    throw nvme_info_discovery_error{"Failed to open: " + filename.string()};
  }
  uint32_t numa_node;
  s >> numa_node;
  return numa_node;
}

std::string nvmeDevPathToModelName(const std::string &devPath) {
  std::filesystem::path filename =
      nvmeNameSpaceToSysfsPciSysfsPath(devPath) / "model";
  std::ifstream s(filename);
  if (!s.is_open()) {
    throw nvme_info_discovery_error{"Failed to open: " + filename.string()};
  }
  std::string model_name((std::istreambuf_iterator<char>(s)),
                         (std::istreambuf_iterator<char>()));
  // don't want a tailing new line here
  model_name.erase(std::remove(model_name.begin(), model_name.end(), '\n'),
                   model_name.end());
  // trim leading and trailing whitespace
  return std::regex_replace(model_name, std::regex("^ +| +$|( ) +"), "$1");
}

uint32_t nvmeDevPathToLinkWidth(const std::string &devPath) {
  std::filesystem::path filename = nvmeNameSpaceToSysfsPciSysfsPath(devPath) /
                                   "device" / "current_link_width";
  std::ifstream s(filename);
  if (!s.is_open()) {
    throw nvme_info_discovery_error{"Failed to open: " + filename.string()};
  }
  uint32_t link_width;
  s >> link_width;
  return link_width;
}

std::string nvmeDevPathToLinkSpeed(const std::string &devPath) {
  std::filesystem::path filename = nvmeNameSpaceToSysfsPciSysfsPath(devPath) /
                                   "device" / "current_link_speed";
  std::ifstream s(filename);
  if (!s.is_open()) {
    throw nvme_info_discovery_error{"Failed to open: " + filename.string()};
  }
  std::string link_speed((std::istreambuf_iterator<char>(s)),
                         (std::istreambuf_iterator<char>()));
  // don't want a tailing new line here
  link_speed.erase(std::remove(link_speed.begin(), link_speed.end(), '\n'),
                   link_speed.end());
  return link_speed;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
extern "C" void numa_error(char *where) { LOG(FATAL) << where; }
extern "C" void numa_warn(int num, char *fmt, ...) { LOG(WARNING) << fmt; }
#pragma clang diagnostic pop

constexpr size_t hugepage = 2 * 1024 * 1024;

static size_t fixSize(size_t bytes) {
  return ((bytes + hugepage - 1) / hugepage) * hugepage;
}

void *topology::cpunumanode::alloc(size_t bytes) const {
  bytes = fixSize(bytes);
  void *mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  LOG_IF(FATAL, mem == MAP_FAILED) << "mmap failed (" << strerror(errno) << ")";
  assert(mem != MAP_FAILED);
  assert((((uintptr_t)mem) % hugepage) == 0);
  linux_run(madvise(mem, bytes, MADV_DONTFORK));
#ifndef NDEBUG
  {
    int status;
    // use move_pages as getCpuNumaNodeAddressed checks only the policy
    auto move_pages_result = move_pages(0, 1, &mem, nullptr, &status, 0);
    if (move_pages_result > 0) {
      LOG(FATAL) << "Failed to move " << move_pages_result << " pages";
    }
    if (move_pages_result < 0) {
      LOG(FATAL) << "Failed to move_pages: " << strerror(errno);
    }
    // check that page has not been prefaulted (status should be -ENOENT)!
    // otherwise, setting the numa policy will not be effective
    assert(status == -ENOENT);
  }
#endif

  // TODO: consider using numa_set_strict
  numa_tonode_memory(mem, bytes, id);

#ifndef NDEBUG
  {
    int status;
    // use move_pages as getCpuNumaNodeAddressed checks only the policy
    assert(move_pages(0, 1, &mem, nullptr, &status, 0) == 0);
    // That check is not critical but only a sanity check, consider removing
    assert(status == -ENOENT);
  }

  if (bytes >= sizeof(int)) {
    // fault first page to check it's allocated on the correct node
    ((int *)mem)[0] = 0;  // SIGBUS on this line means we run out of huge pafes
                          // in that CPU numa node
    // check using the policy
    assert(topology::getInstance().getCpuNumaNodeAddressed(mem) == this);

    {
      // now the first page should have been prefaulted, verify using move_pages
      int status;
      assert(move_pages(0, 1, &mem, nullptr, &status, 0) == 0);
      // the faulted page should be on the correct socket now
      assert(status == id);
    }
  }
#endif

  return mem;
  //  return numa_alloc_onnode(bytes, id);
}

void topology::cpunumanode::free(void *mem, size_t bytes) {
  // numa_free(mem, bytes);
  bytes = fixSize(bytes);
  linux_run(munmap(mem, bytes));
}

size_t topology::cpunumanode::getMemorySize() const {
  return numa_node_size64(id, nullptr);
}

void topology::init() {
  instance.init_();
  std::cout << topology::getInstance() << std::endl;
}

static uint64_t profilingStartTimestamp = 0;

class CUPTILogger {
 public:
  struct entry {
    /* Relative to profilingStartTimestamp */
    uint64_t start_ns;
    uint64_t end_ns;
    std::string kernel_name;
    uint32_t dev;
  };

 private:
  std::deque<entry> log;

 public:
  ~CUPTILogger() {
    std::ofstream out{"cupti.csv"};
    out << "start,end,kernel_name\n";
    for (auto &e : log) {
      out << e.start_ns << ',' << e.end_ns << ',' << e.kernel_name << "_("
          << e.dev << ")" << '\n';
    }
  }

 public:
  void push(entry s) { log.emplace_back(std::move(s)); }
};

template <typename T>
class Monitor {
  std::mutex m;
  T obj;

 public:
  template <typename F>
  auto withLockDo(const F &f) {
    std::unique_lock<std::mutex> lock{m};
    return f(obj);
  }
};

static Monitor<CUPTILogger> cuptiLogger;

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) noexcept {
  if (validSize > 0) {
    cuptiLogger.withLockDo([&](auto &log) {
      CUpti_Activity *record = nullptr;
      while (true) {
        auto status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
          switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_INVALID:
              break;
// Added in cuda 11.7
#if (CUDART_VERSION >= 11070)
            case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
              break;
#endif
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
              CUpti_ActivityMemcpy4 *kernel = (CUpti_ActivityMemcpy4 *)record;
              if (kernel->start == 0 && kernel->end == 0) {
                LOG(WARNING) << "Ignoring memcpy event without time info";
              } else {
                log.push({.start_ns = kernel->start - profilingStartTimestamp,
                          .end_ns = kernel->end - profilingStartTimestamp,
                          .kernel_name = "memcpy",
                          .dev = kernel->deviceId});
              }
              break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET:
              break;
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
              const char *kindString =
                  (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL"
                                                               : "CONC KERNEL";
              CUpti_ActivityKernel3 *kernel = (CUpti_ActivityKernel3 *)record;
              if (kernel->start == 0 && kernel->end == 0) {
                LOG(WARNING)
                    << "Ignoring " << kindString << " event without time info";
              } else {
                log.push({.start_ns = kernel->start - profilingStartTimestamp,
                          .end_ns = kernel->end - profilingStartTimestamp,
                          .kernel_name = kernel->name,
                          .dev = kernel->deviceId});
                //  LOG(WARNING)
                //      << kindString << '"' << kernel->name << '"' << " [ "
                //      << kernel->start << " - " << kernel->end << " ] "
                //      << kernel->deviceId << " ctx=" << kernel->contextId
                //      << ", strm=" << kernel->streamId
                //      << ", corr=" << kernel->correlationId;
              }
              break;
            }
            case CUPTI_ACTIVITY_KIND_DRIVER:
              break;
            case CUPTI_ACTIVITY_KIND_RUNTIME:
              break;
            case CUPTI_ACTIVITY_KIND_EVENT:
              break;
            case CUPTI_ACTIVITY_KIND_METRIC:
              break;
            case CUPTI_ACTIVITY_KIND_DEVICE:
              break;
            case CUPTI_ACTIVITY_KIND_CONTEXT:
              break;
            case CUPTI_ACTIVITY_KIND_NAME:
              break;
            case CUPTI_ACTIVITY_KIND_MARKER:
              break;
            case CUPTI_ACTIVITY_KIND_MARKER_DATA:
              break;
            case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
              break;
            case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS:
              break;
            case CUPTI_ACTIVITY_KIND_BRANCH:
              break;
            case CUPTI_ACTIVITY_KIND_OVERHEAD:
              break;
            case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
              break;
            case CUPTI_ACTIVITY_KIND_PREEMPTION:
              break;
            case CUPTI_ACTIVITY_KIND_ENVIRONMENT:
              break;
            case CUPTI_ACTIVITY_KIND_EVENT_INSTANCE:
              break;
            case CUPTI_ACTIVITY_KIND_MEMCPY2:
              break;
            case CUPTI_ACTIVITY_KIND_METRIC_INSTANCE:
              break;
            case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION:
              break;
            case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
              break;
            case CUPTI_ACTIVITY_KIND_FUNCTION:
              break;
            case CUPTI_ACTIVITY_KIND_MODULE:
              break;
            case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
              break;
            case CUPTI_ACTIVITY_KIND_SHARED_ACCESS:
              break;
            case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
              break;
            case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
              break;
            case CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION:
              break;
            case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
              break;
            case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
              break;
            case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
              break;
            case CUPTI_ACTIVITY_KIND_CUDA_EVENT:
              break;
            case CUPTI_ACTIVITY_KIND_STREAM:
              break;
            case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
              break;
            case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
              break;
            case CUPTI_ACTIVITY_KIND_NVLINK:
              break;
            case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT:
              break;
            case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE:
              break;
            case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC:
              break;
            case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE:
              break;
            case CUPTI_ACTIVITY_KIND_MEMORY: {
              CUpti_ActivityMemory2 *kernel = (CUpti_ActivityMemory2 *)record;
              log.push(
                  {.start_ns = kernel->timestamp - profilingStartTimestamp,
                   .end_ns = kernel->timestamp + 1 - profilingStartTimestamp,
                   .kernel_name = kernel->name,
                   .dev = kernel->deviceId});
              break;
            }
            case CUPTI_ACTIVITY_KIND_PCIE:
              break;
            case CUPTI_ACTIVITY_KIND_OPENMP:
              break;
            case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API:
              break;
            case CUPTI_ACTIVITY_KIND_MEMORY2:
              break;
            case CUPTI_ACTIVITY_KIND_MEMORY_POOL:
              break;
            case CUPTI_ACTIVITY_KIND_COUNT:
              break;
            case CUPTI_ACTIVITY_KIND_FORCE_INT:
              break;
          }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
          break;
        else {
          LOG(ERROR) << "error";
          exit(-1);
          //        gpu_run(status);
        }
      }

      // report any records dropped from the queue
      size_t dropped = 0;
      cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
      if (dropped != 0) {
        LOG(WARNING) << "Dropped " << dropped << " activity records";
      }
    });
  }

  free(buffer);
  //  BlockManager::release_buffer(buffer);
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *)malloc(1024 * 1024 + 8);
  if (bfr == nullptr) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = 1024 * 1024;
#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))
  *buffer = ALIGN_BUFFER(bfr, 8);
  *maxNumRecords = 0;
}

void topology::init_() {
  // Check if topology is already initialized and if yes, return early
  // This should only happen when a unit-test is reinitializing proteus but it
  // should not happen in normal execution, except if we "restart" proteus
  if (!core_info.empty()) {
#ifndef NDEBUG
    auto core_cnt = sysconf(_SC_NPROCESSORS_ONLN);
    assert(core_info.size() == core_cnt);
#endif
    return;
  }
  assert(cpu_info.empty() && "Is topology already initialized?");
  assert(core_info.empty() && "Is topology already initialized?");
  unsigned int gpus = 0;
  auto nvml_res = nvmlInit();
  if (nvml_res == NVML_SUCCESS) {
    // We can not use gpu_run(...) before we set gpu_cnt, call gpuAssert
    // directly.
    auto ret = cudaGetDeviceCount((int *)&gpus);
    if (ret == cudaErrorNoDevice) {
      // Still not an error if we do not have any devices
      gpus = 0;
    } else {
      gpuAssert(ret, __FILE__, __LINE__);
    }
  }
  gpu_cnt = gpus;

  // Creating gpunodes requires that we know the number of cores,
  // so start by reading the CPU configuration
  core_cnt = sysconf(_SC_NPROCESSORS_ONLN);
  assert(core_cnt > 0);

  std::map<uint32_t, std::vector<uint32_t>> numa_to_cores_mapping;

  for (uint32_t j = 0; j < core_cnt; ++j) {
    numa_to_cores_mapping[numa_node_of_cpu(j)].emplace_back(j);
  }

  uint32_t max_numa_id = 0;
  for (const auto &numa : numa_to_cores_mapping) {
    cpu_info.emplace_back(numa.first, numa.second, cpu_info.size());
    max_numa_id = std::max(max_numa_id, cpu_info.back().id);
  }

  for (auto &n : cpu_info) {
    for (const auto &n2 : cpu_info) {
      n.distance.emplace_back(numa_distance(n.id, n2.id));
    }
  }

  cpunuma_index.resize(max_numa_id + 1);
  for (auto &ci : cpunuma_index) ci = 0;

  for (size_t i = 0; i < cpu_info.size(); ++i) {
    cpunuma_index[cpu_info[i].id] = i;
  }

  cpucore_index.resize(core_cnt);
  for (const auto &cpu : cpu_info) {
    for (const auto &core : cpu.local_cores) {
      cpucore_index[core] = core_info.size();
      core_info.emplace_back(core, cpu.id, core_info.size(), cpu.index_in_topo,
                             ThreadSiblingParser::getThreadSiblings(core));
    }
  }

  assert(core_info.size() == core_cnt);

  // Now create the GPU nodes
  for (uint32_t i = 0; i < gpu_cnt; ++i) {
    gpu_info.emplace_back(i, i, core_info);
    const auto &ind = cpunuma_index[gpu_info.back().local_cpu_id];
    cpu_info[ind].local_gpus.push_back(i);
  }

  if (/* DISABLES CODE */ (false)) {
    size_t attrValue = 1024 * 1024;  // BlockManager::block_size;
    size_t attrValueSize = sizeof(size_t);
    size_t poolLimit = 20;

    gpu_run(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    gpu_run(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    gpu_run(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    gpu_run(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
    gpu_run(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                      &attrValueSize, &attrValue));
    gpu_run(
        cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                  &attrValueSize, &poolLimit));

    {
      event_range<range_log_op::CUPTI_GET_START_TIMESTAMP> ev{this};
      gpu_run(cuptiGetTimestamp(&profilingStartTimestamp));
    }

    // Register callbacks for buffer requests and for buffers completed by
    // CUPTI.
    gpu_run(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  }

  // warm-up GPUs
  for (const auto &gpu : gpu_info) {
    gpu_run(cudaSetDevice(gpu.id));
    gpu_run(cudaFree(nullptr));
  }

  // P2P check & enable
  for (auto &gpu : gpu_info) {
    gpu.connectivity.resize(gpu_cnt);

    gpu_run(cudaSetDevice(gpu.id));
    // set_device_on_scope d(gpu);
    for (const auto &gpu2 : gpu_info) {
      if (gpu2.id != gpu.id) {
        int t = 0;
        gpu_run(cudaDeviceCanAccessPeer(&t, gpu.id, gpu2.id));
        if (t) {
          gpu_run(cudaDeviceEnablePeerAccess(gpu2.id, 0));
        } else {
          std::cout << "Warning: P2P disabled for : GPU-" << gpu.id;
          std::cout << " -> GPU-" << gpu2.id << std::endl;
        }

        gpu_run(nvmlDeviceGetTopologyCommonAncestor(
            gpu.handle, gpu2.handle, &(gpu.connectivity[gpu2.id])));
      }
    }
  }

  // collect infiniband info
  ib_info = ib::discover();

  // collect NVMe info
  init_nvmeStorage();
}

/**
 *
 * @return an array of strings corresponding to all nvme namespace device paths
 * on this machine e.g ["/dev/nvme0n1"]
 *
 * see https://unix.stackexchange.com/a/452135 for nvme device naming
 * conventions
 */
std::vector<std::string> discoverNvmeDevicePaths() {
  // Using 'sudo nvme list -o json' would be nice, except that it 1. require
  // sudo, and 2. is very dependent on ubuntu version for the info returned.
  // older versions of ubuntu the only useful info we get is the device name
  // Using libnvme would also be nice, but is also not as portable for the same
  // reasons
  std::vector<std::string> devicePaths{};
  for (const auto &device : std::filesystem::directory_iterator("/dev/")) {
    const auto devString = device.path().string();
    std::smatch nvme_device_match;
    if (std::regex_search(devString, nvme_device_match,
                          std::regex("\\/dev\\/nvme[0-9]+n[0-9]+$"))) {
      devicePaths.push_back(device.path().string());
    }
  }

  return devicePaths;
}

/**
 * Find NVMe drives on this machine by parsing /dev/nvme*
 * and then additional information on the drives by parsing sysfs
 * Populates nvmeStorage_info
 */
void topology::init_nvmeStorage() {
  auto nvmeDevPaths = discoverNvmeDevicePaths();
  for (size_t i = 0; i < nvmeDevPaths.size(); i++) {
    try {
      nvmeStorage_info.emplace_back(nvmeDevPaths.at(i), i);
    } catch (nvme_info_discovery_error &e) {
      LOG(WARNING) << "Ignoring nvme device: " << e.what()
                   << ". Is this a valid nvme device?";
    }
  }
}

topology::topology() {}

std::ostream &operator<<(std::ostream &out, const cpu_set_t &cpus) {
  long cores = sysconf(_SC_NPROCESSORS_ONLN);

  bool printed = false;

  for (int i = 0; i < cores; ++i)
    if (CPU_ISSET(i, &cpus)) {
      if (printed) out << ",";
      printed = true;
      out << i;
    }

  return out;
}

std::ostream &operator<<(std::ostream &out, const topology &topo) {
  out << "numa nodes: " << topo.getCpuNumaNodeCount() << "\n";
  out << "core count: " << topo.getCoreCount() << "\n";
  out << "gpu  count: " << topo.getGpuCount() << "\n";
  out << "IB   count: " << topo.getIBCount() << "\n";
  out << "NVMe count: " << topo.getNvmeCount() << "\n";

  out << '\n';

  char core_mask[topo.core_cnt + 1];
  core_mask[topo.core_cnt] = '\0';

  auto digits = (uint32_t)std::ceil(std::log10(topo.core_cnt));

  for (uint32_t k = digits; k > 0; --k) {
    uint32_t base = std::pow(10, k - 1);

    if (k == ((digits + 1) / 2))
      out << "core: ";
    else
      out << "      ";

    if (1 == digits)
      out << ' ';
    else if (k == digits)
      out << '/';
    else if (k == 1)
      out << '\\';
    else
      out << '|';
    out << std::setw(base + 4 + 4 + 3 + 18) << ((k == 1) ? '0' : ' ');

    for (uint32_t i = base; i < topo.core_cnt; ++i) {
      out << (i / base) % 10;
    }
    out << '\n';
  }

  for (const auto &node : topo.getCpuNumaNodes()) {
    out << "node: " << std::setw(6) << node.id << " | ";

    out << std::setw(4 + 4 + 3) << ' ' << " | ";

    out << "cores: ";

    // for ( auto cpu_id : node.logical_cpus) {
    //     out << std::setw(4) << cpu_id << " ";
    // }

    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';

    for (auto cpu_id : node.local_cores) core_mask[cpu_id] = 'x';

    out << core_mask << '\n';
  }

  out << '\n';

  for (const auto &gpu : topo.getGpus()) {
    unsigned int nvml_ind = 0;
    gpu_run(nvmlDeviceGetIndex(gpu.handle, &nvml_ind));
    out << "gpu : " << std::setw(2) << gpu.id;
    out << std::setw(4) << ("(" + std::to_string(nvml_ind) + ")") << " | ";
    out << "node : " << std::setw(4) << gpu.local_cpu_id << " | ";
    out << "cores: ";

    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';

    for (auto cpu_id : gpu.local_cores) core_mask[cpu_id] = 'x';

    bytes memBusWidth{static_cast<size_t>(gpu.properties.memoryBusWidth / 8)};
    out << core_mask;
    out << " | memory bus width: " << memBusWidth;
    out << " | name: " << gpu.properties.name;
    out << '\n';
    // for ( auto cpu_id : gpu.local_cores  ) {
    //     if (cpu_id)
    //     out << std::setw(4) << cpu_id << " ";
    // }
    // out << '\n';
  }

  out << '\n';

  for (const auto &ib : topo.getIBs()) {
    out << "ib:   " << std::setw(6) << ib.id << " | ";

    const auto &numanode = topo.findLocalCPUNumaNode(ib);
    out << "node : " << std::setw(4) << numanode.id << " | ";
    out << "cores: ";

    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';

    for (auto cpu_id : numanode.local_cores) {
      core_mask[cpu_id] = 'x';
    }

    out << core_mask;
    out << " | name: " << ib;
    out << '\n';
  }

  // size_t sockets = topo.cpu_info.size();

  out << '\n';

  for (const auto &node : topo.getCpuNumaNodes()) {
    out << "node: ";
    out << node.id << " | ";
    for (auto d : node.distance) out << std::setw(4) << d;
    out << '\n';
  }

  out << '\n';

  for (const auto &gpu : topo.getGpus()) {
    out << "gpu : " << gpu.id << " | ";
    for (auto d : gpu.connectivity) out << std::setw(4) << d;
    out << '\n';
  }

  for (const auto &nvme : topo.getNvmes()) {
    out << "nvme: " << nvme.index_in_topo << " | ";
    out << "node: " << nvme.local_cpu_id << " | ";

    const auto &numanode = nvme.getLocalCPUNumaNode();
    out << "cores: ";

    // clear mask
    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';
    // set mask
    for (auto cpu_id : numanode.local_cores) {
      core_mask[cpu_id] = 'x';
    }
    out << core_mask;

    out << " |  " << nvme;
    out << "\n";
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const topology::nvmeStorage &nvme) {
  out << "model name: " << nvme.model_name
      << ", link_speed: " << nvme.link_speed
      << ", PCIe lanes: " << nvme.link_width
      << ", device path: " << nvme.devPath;
  return out;
}

topology::gpunode::gpunode(uint32_t id, uint32_t index_in_topo,
                           const std::vector<topology::core> &all_cores,
                           topologyonly_construction)
    : id(id), handle(getGPUHandle(id)), index_in_topo(index_in_topo) {
#ifndef NCUDA
  gpu_run(cudaGetDeviceProperties(&properties, id));

  defaultGridDim =
      dim3(std::max((decltype(dim3::x))properties.multiProcessorCount *
                        ((properties.maxThreadsPerMultiProcessor +
                          defaultBlockDim.x - 1) /
                         defaultBlockDim.x),
                    defaultGridDim.x),
           1, 1);

  uint32_t sets = ((all_cores.size() + 63) / 64);
  uint64_t cpuSet[sets];
  for (uint32_t i = 0; i < sets; ++i) cpuSet[i] = 0;

  CPU_ZERO(&local_cpu_set);

  gpu_run(nvmlDeviceGetCpuAffinity(handle, sets, cpuSet));
  for (uint32_t i = 0; i < sets; ++i) {
    for (uint32_t k = 0; k < 64; ++k) {
      if ((cpuSet[i] >> k) & 1) CPU_SET(64 * i + k, &(local_cpu_set));
    }
  }

  uint32_t invalid = ~((uint32_t)0);
  uint32_t tmp_cpu = invalid;

  for (const auto &c : all_cores) {
    if (CPU_ISSET(c.id, &(local_cpu_set))) {
      local_cores.push_back(c.id);

      uint32_t cpu = c.local_cpu_id;
      assert(tmp_cpu == invalid || tmp_cpu == cpu);
      tmp_cpu = cpu;
    }
  }

  assert(tmp_cpu != invalid);
  local_cpu_id = tmp_cpu;
#else
  assert(false);
#endif
}

size_t topology::gpunode::getMemorySize() const {
  return properties.totalGlobalMem;
}

topology::cpunumanode::cpunumanode(uint32_t id,
                                   const std::vector<uint32_t> &local_cores,
                                   uint32_t index_in_topo,
                                   topologyonly_construction)
    : id(id),
      // distance(b.distance),
      index_in_topo(index_in_topo),
      local_cores(local_cores) {
  CPU_ZERO(&local_cpu_set);
  for (const auto &c : local_cores) CPU_SET(c, &local_cpu_set);
}

topology::nvmeStorage::nvmeStorage(const std::string &devPath,
                                   uint32_t index_in_topo,
                                   // do not remove argument!!!
                                   topologyonly_construction)
    : devPath(devPath),
      id(index_in_topo),
      index_in_topo(index_in_topo),
      local_cpu_id(nvmeDevPathToNumaNodeId(devPath)),
      link_speed(nvmeDevPathToLinkSpeed(devPath)),
      link_width(nvmeDevPathToLinkWidth(devPath)),
      model_name(nvmeDevPathToModelName(devPath)){};

nvmlDevice_t topology::gpunode::getGPUHandle(unsigned int id) {
  cudaDeviceProp prop;
  gpu_run(cudaGetDeviceProperties(&prop, id));

  // NVML ignores CUDA_VISIBLE_DEVICES env variable, so we have to go over
  // all the available devices and find the one corresponding to the
  // one referenced by @p id for the runtime api.
  // And, "yes, it's expected", source:
  // https://devtalk.nvidia.com/default/topic/815835/different-index-definition-in-nvml-amp-cuda-runtime-/

  unsigned int nvml_count = 0;
  gpu_run(nvmlDeviceGetCount(&nvml_count));
  // assert(device_count == gpus &&
  //        "NMVL disagrees with cuda about the number of GPUs");

  // source:
  // https://devblogs.nvidia.com/increase-performance-gpu-boost-k80-autoboost/
  for (unsigned int nvml_ind = 0; nvml_ind < nvml_count; ++nvml_ind) {
    nvmlDevice_t d{};
    gpu_run(nvmlDeviceGetHandleByIndex(nvml_ind, &d));

    nvmlPciInfo_t pcie_info;
    gpu_run(nvmlDeviceGetPciInfo(d, &pcie_info));

    if (static_cast<unsigned int>(prop.pciBusID) == pcie_info.bus &&
        static_cast<unsigned int>(prop.pciDeviceID) == pcie_info.device &&
        static_cast<unsigned int>(prop.pciDomainID) == pcie_info.domain) {
      return d;
    }
  }
  throw std::runtime_error("failed to locate device in nvml!");
}

topology topology::instance;

extern "C" int get_rand_core_local_to_ptr(const void *p) {
  // const auto *dev = topology::getInstance().getGpuAddressed(p);
  // if (dev) return dev->local_cores[rand() % dev->local_cores.size()];
  // const auto *cpu = topology::getInstance().getCpuNumaNodeAddressed(p);
  // return cpu->local_cores[rand() % cpu->local_cores.size()];

  // actually, for the current exchange implementation we should return
  // the integer i such that (i % #gpus) is a _gpu_ local to the current
  // numa node addressed. (and yes, this will cause problems on machines
  // without GPUs, but such machines need issue #16 to be resolved)
  // FIXME: related to issue #16 and the above comment
  // FIXME: *up*

  const auto &topo = topology::getInstance();
  const auto gpu_count = topo.getGpuCount();
  const auto *dev = topology::getInstance().getGpuAddressed(p);
  if (dev) return dev->id + ((rand() / gpu_count) * gpu_count);

  const auto *cpu = topology::getInstance().getCpuNumaNodeAddressed(p);

  const auto &local_gpus = cpu->local_gpus;
  size_t local_gpu_count = local_gpus.size();
  if (local_gpu_count == 0) return rand();

  const auto &sdev = local_gpus[rand() % local_gpu_count];
  return sdev + ((rand() / gpu_count) * gpu_count);
}

extern "C" int rand_local_cpu(const void *p, uint64_t fanout) {
  const auto *g = topology::getInstance().getGpuAddressed(p);
  if (g) assert(false && "TODO");
  const auto *c = topology::getInstance().getCpuNumaNodeAddressed(p);
  assert(c);
  size_t socket = c->index_in_topo;
  size_t nsockets = topology::getInstance().getCpuNumaNodeCount();
  size_t ulimit = (fanout - 1 - socket) / nsockets;
  size_t r = rand() % ulimit;
  return socket + r * nsockets;
}

const topology::cpunumanode &topology::gpunode::getLocalCPUNumaNode() const {
  return topology::getInstance().getCpuNumaNodeById(local_cpu_id);
}

const topology::cpunumanode &topology::nvmeStorage::getLocalCPUNumaNode()
    const {
  return topology::getInstance().getCpuNumaNodeById(local_cpu_id);
}

const topology::cpunumanode &topology::core::getLocalCPUNumaNode() const {
  return topology::getInstance().getCpuNumaNodeById(local_cpu_id);
}

set_exec_location_on_scope topology::cpunumanode::set_on_scope() const {
  return {*this};
}

set_exec_location_on_scope topology::core::set_on_scope() const {
  return {*this};
}

set_exec_location_on_scope topology::gpunode::set_on_scope() const {
  return {*this};
}

set_exec_location_on_scope topology::nvmeStorage::set_on_scope() const {
  return {*this};
}
