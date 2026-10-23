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

#include <gtest/gtest.h>

#include <codegen/expressions/expressionTypes.hpp>
#include <cstring>
#include <fstream>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/operators/relbuilder.hpp>
#include <olap/plan/query-result.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/common/common.hpp>
#include <platform/threadpool/threadpool.hpp>

#include "lib/operators/mem-move/compression.hpp"
#include "utils.hpp"

// `SCOPED_TRACE("")` raises this error
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"

using namespace dangling_attr;

TEST(NvmePluginAttributePartMetaDataTest, from_file_small) {
  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  NvmePlugin::AttributePartMetaData x{md_path};
  EXPECT_EQ(x.num_blocks, 6);
  EXPECT_EQ(x.block_sizes.size(), 6);
  EXPECT_EQ(x.block_offsets.size(), 6);
  EXPECT_EQ(x.block_sizes.front(), 2097152);
  EXPECT_EQ(x.block_sizes.back(), 1514240);
  EXPECT_EQ(x.block_offsets.front(), 0);
  EXPECT_EQ(x.block_offsets.back(), 10485760);
  EXPECT_EQ(x.value_counts.front(), 524288);
  EXPECT_EQ(x.value_counts.back(), 378560);
  EXPECT_EQ(x.data_format,
            NvmePlugin::AttributePartMetaData::DataFormat_t::UNCOMPRESSED);
}

TEST(NvmePluginAttributePartMetaDataTest, from_file_small_compressed) {
  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/"
      "ssb100_compressed_date.csv.d_datekey.metadata.json";
  NvmePlugin::AttributePartMetaData x{md_path};
  EXPECT_EQ(x.num_blocks, 1);
  EXPECT_EQ(x.block_sizes.size(), 1);
  EXPECT_EQ(x.block_offsets.size(), 1);
  EXPECT_EQ(x.block_sizes.front(), 10266);
  EXPECT_EQ(x.block_offsets.front(), 0);
  EXPECT_EQ(x.value_counts.front(), 2556);
  EXPECT_EQ(x.chunk_sizes.front().front(), 10266);
  EXPECT_EQ(x.max_compressed_block_size, 10266);
  EXPECT_EQ(x.decompressed_chunk_size, 16384);
  EXPECT_EQ(x.data_format,
            NvmePlugin::AttributePartMetaData::DataFormat_t::COMPRESSED);
}

TEST(PageId_t, ptr_tagging) {
  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path, md_path};
  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));
  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);
  auto context = OlapParallelContext("test");
  NvmePlugin testPlugin(&context, relation_md);

  void* page_id_as_ptr = getNvmePageIdPtr(&testPlugin, 0, 0, 0);

  EXPECT_TRUE(NvmePlugin::PageId_t::isPageIdPtr(page_id_as_ptr));
  constexpr uintptr_t full_mask = 1ULL << 63;
  static_assert(sizeof(uintptr_t) == 8, "uintptr_t is not 8 bytes");
  EXPECT_TRUE(reinterpret_cast<uintptr_t>(page_id_as_ptr) & full_mask);

  auto page_id_from_ptr = NvmePlugin::PageId_t::from_ptr(page_id_as_ptr);

  constexpr int8_t numa_mask = 1u << 7;
  EXPECT_FALSE(
      reinterpret_cast<uint8_t>(page_id_from_ptr.getCpuNumaAffinity()) &
      numa_mask);
}

TEST(PageId_t, invertable) {
  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path, md_path};
  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));
  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);
  auto context = OlapParallelContext("test");
  NvmePlugin testPlugin(&context, relation_md);

  const uint8_t cpu_numa_affinity =
      1;  /// can't test since it depends on affinity of the test file location
  const uint8_t expected_attribute_no = 0;
  const uint8_t expected_partition_no = 0;
  const uint32_t expected_block_no = 1;

  NvmePlugin::PageId_t from_cpp_cons{cpu_numa_affinity, expected_attribute_no,
                                     expected_partition_no, expected_block_no};
  void* page_id_as_ptr = getNvmePageIdPtr(
      &testPlugin, from_cpp_cons.getAttributeNo(),
      from_cpp_cons.getPartitionNo(), from_cpp_cons.getBlockNo());
  NvmePlugin::PageId_t y = NvmePlugin::PageId_t::from_ptr(page_id_as_ptr);
  EXPECT_EQ(from_cpp_cons.getAttributeNo(), y.getAttributeNo());
  EXPECT_EQ(from_cpp_cons.getPartitionNo(), y.getPartitionNo());
  EXPECT_EQ(from_cpp_cons.getBlockNo(), y.getBlockNo());
}

TEST(DecompressionTest, decompress_datekey_gpu) {
  if (topology::getInstance().getGpuCount() < 1) {
    GTEST_SKIP() << "No GPUs available";
  }

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/"
      "ssb100_compressed_date.csv.d_datekey.metadata.json";
  NvmePlugin::AttributePartMetaData partMetaData(md_path);
  EXPECT_EQ(partMetaData.data_format,
            NvmePlugin::AttributePartMetaData::DataFormat_t::COMPRESSED);
  EXPECT_EQ(partMetaData.num_blocks, 1);

  auto read_size = partMetaData.block_sizes[0];
  if (read_size % 512 != 0) {
    read_size += 512 - (read_size % 512);  // align to 512 bytes for O_DIRECT
  }
  char* compressed_buf =
      static_cast<char*>(std::aligned_alloc(4096, read_size));
  PCHECK(read(partMetaData.fd, compressed_buf, read_size) > 0);

  auto gpu_decompressor = GpuDecompressor(partMetaData.decompressed_chunk_size,
                                          1, partMetaData.chunk_sizes[0].size(),
                                          CompressionAlgorithm::LZ4);

  cudaStream_t stream = nullptr;
  gpu_run(cudaStreamCreate(&stream));

  void* compressed_buf_gpu = nullptr;
  gpu_run(cudaMallocAsync(&compressed_buf_gpu, partMetaData.block_sizes[0],
                          stream));
  gpu_run(cudaMemcpyAsync(compressed_buf_gpu, compressed_buf,
                          partMetaData.block_sizes[0], cudaMemcpyHostToDevice,
                          stream));

  void* decompressed_buf_gpu = nullptr;
  size_t decomp_size = partMetaData.value_counts[0] * sizeof(int32_t);
  gpu_run(cudaMallocAsync(&decompressed_buf_gpu, decomp_size, stream));

  auto compress_span_gpu = std::span<char>(
      static_cast<char*>(compressed_buf_gpu), partMetaData.block_sizes[0]);
  auto decomp_span_gpu =
      std::span<char>(static_cast<char*>(decompressed_buf_gpu), decomp_size);

  ASSERT_GE(gpu_decompressor.decompress_block_gpu(partMetaData.chunk_sizes[0],
                                                  compress_span_gpu,
                                                  decomp_span_gpu, stream),
            0)
      << "decompression launch failed";
  cudaStreamSynchronize(stream);
  ASSERT_EQ(gpu_decompressor.get_last_batch_bytes_decompressed(),
            partMetaData.value_counts[0] * sizeof(int32_t))
      << "decompression failed";

  /* Validate against input data.  */
  // Transfer the decompressed buffer to the host.
  char* decompress_buf = new char[decomp_size];
  gpu_run(cudaMemcpy(decompress_buf, decomp_span_gpu.data(), decomp_size,
                     cudaMemcpyDeviceToHost));

  auto original_data = mmap_file("inputs/ssbm100/date.csv.d_datekey", PINNED);

  EXPECT_EQ(decomp_size, original_data.getFileSize());

  EXPECT_EQ(memcmp(original_data.getData(), (void*)decompress_buf, decomp_size),
            0);
}

TEST(DecompressionTest, decompress_gpu_batch) {
  if (topology::getInstance().getGpuCount() < 1) {
    GTEST_SKIP() << "No GPUs available";
  }
  auto& gpu = topology::getInstance().getGpus()[0];
  auto exec_scope = gpu.set_on_scope();
  // FIXME server specific path
  const std::filesystem::path md_path =
      "/nvme14/nicholso/data/compressed_ssbm1000/"
      "supplier.csv.s_region_0_1.metadata.json";
  NvmePlugin::AttributePartMetaData partMetaData(md_path);
  EXPECT_EQ(partMetaData.data_format,
            NvmePlugin::AttributePartMetaData::DataFormat_t::COMPRESSED);
  const int batch_size = 2;
  EXPECT_GE(partMetaData.num_blocks, batch_size)
      << "not a hard correctness requirement, but this test is mean to test "
         "batch decompression";

  std::vector<proteus::managed_ptr> uncompressed_blocks{};
  uncompressed_blocks.reserve(partMetaData.num_blocks);

  auto compressed_data_gpu =
      mmap_file(partMetaData.data_file_path, data_loc::GPU_RESIDENT);

  std::vector<const char*> compressed_block_ptrs(partMetaData.num_blocks);
  for (int i = 0; i < partMetaData.num_blocks; i++) {
    auto read_size = partMetaData.block_sizes[i];
    if (read_size % 512 != 0) {
      read_size += 512 - (read_size % 512);  // align to 512 bytes for O_DIRECT
    }
    compressed_block_ptrs[i] =
        static_cast<const char*>(compressed_data_gpu.getData()) +
        partMetaData.block_offsets[i];
    uncompressed_blocks.emplace_back(
        BlockManager::h_get_buffer(gpu.index_in_topo));
  }

  // assuming the first block will have the maximum number of chunks per block.
  const size_t chunks_per_block = partMetaData.chunk_sizes[0].size();
  EXPECT_GT(chunks_per_block, 0);
  auto gpu_decompressor =
      GpuDecompressor(partMetaData.decompressed_chunk_size, batch_size,
                      chunks_per_block, CompressionAlgorithm::LZ4);

  cudaStream_t stream = createNonBlockingStream();

  for (int i = 0; i < partMetaData.num_blocks; i += batch_size) {
    std::vector<std::vector<uint32_t>> block_chunk_sizes;
    std::vector<std::span<char>> block_compressed_buffers;
    std::vector<std::span<char>> block_output_buffers;
    size_t expected_batch_decomp_size_bytes = 0;
    for (int j = i; (j < i + batch_size) && (j < partMetaData.num_blocks);
         j++) {
      block_chunk_sizes.emplace_back(partMetaData.chunk_sizes[j]);

      const size_t decomp_size = partMetaData.value_counts[j] * sizeof(int32_t);
      expected_batch_decomp_size_bytes += decomp_size;
      auto compress_span_gpu =
          std::span<char>(const_cast<char*>(compressed_block_ptrs[j]),
                          partMetaData.block_sizes[j]);
      auto decomp_span_gpu = std::span<char>(
          static_cast<char*>(uncompressed_blocks[j].get()), decomp_size);
      block_compressed_buffers.emplace_back(compress_span_gpu);
      block_output_buffers.emplace_back(decomp_span_gpu);
    }

    auto submit_success = gpu_decompressor.batch_decompress_block_gpu(
        block_chunk_sizes, block_compressed_buffers, block_output_buffers,
        stream);
    ASSERT_EQ(submit_success, 0);

    // wait for the batch to complete
    cudaStreamSynchronize(stream);
    auto bytes_decompressed =
        gpu_decompressor.get_last_batch_bytes_decompressed();
    ASSERT_GE(bytes_decompressed, 0)
        << "failed to decompression chunk: " << -i
        << " which is chunk: " << (-i) % chunks_per_block
        << " in block: " << (-i) / chunks_per_block
        << " this occurred in batch " << i;
    ASSERT_EQ(expected_batch_decomp_size_bytes, bytes_decompressed);
  }

  std::vector<proteus::managed_ptr> uncompressed_blocks_cpu{};
  uncompressed_blocks_cpu.reserve(partMetaData.num_blocks);
  for (int i = 0; i < partMetaData.num_blocks; i++) {
    uncompressed_blocks_cpu.emplace_back(BlockManager::get_buffer());
    BlockManager::overwrite_bytes(
        uncompressed_blocks_cpu[i].get(), uncompressed_blocks[i].get(),
        partMetaData.value_counts[i] * sizeof(int32_t), stream, false);
  }

  validateLoadedBlocks(uncompressed_blocks_cpu,
                       "inputs/ssbm1000/supplier.csv.s_region");

  for (int i = 0; i < partMetaData.num_blocks; i++) {
    BlockManager::release_buffer(uncompressed_blocks_cpu[i].release());
    BlockManager::release_buffer(uncompressed_blocks[i].release());
  }
}

TEST(DecompressionTest, decompress_datekey) {
  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/"
      "ssb100_compressed_date.csv.d_datekey.metadata.json";
  NvmePlugin::AttributePartMetaData partMetaData(md_path);
  EXPECT_EQ(partMetaData.data_format,
            NvmePlugin::AttributePartMetaData::DataFormat_t::COMPRESSED);
  EXPECT_EQ(partMetaData.num_blocks, 1);

  auto read_size = partMetaData.block_sizes[0];
  if (read_size % 512 != 0) {
    read_size += 512 - (read_size % 512);  // align to 512 bytes for O_DIRECT
  }
  char* compressed_buf =
      static_cast<char*>(std::aligned_alloc(4096, read_size));
  PCHECK(read(partMetaData.fd, compressed_buf, read_size) > 0);

  auto decompressed_buf =
      static_cast<char*>(std::aligned_alloc(4096, BlockManager::block_size));
  auto decomp_span =
      std::span<char>(decompressed_buf, BlockManager::block_size);
  auto comp_span = std::span<char>(compressed_buf, partMetaData.block_sizes[0]);
  ASSERT_EQ(decompress_block(partMetaData.chunk_sizes[0], comp_span,
                             decomp_span, partMetaData.decompressed_chunk_size),
            partMetaData.value_counts[0] * sizeof(int32_t))
      << "decompression failed";

  auto original_data = mmap_file("inputs/ssbm100/date.csv.d_datekey", PINNED);

  EXPECT_EQ(memcmp(original_data.getData(), decompressed_buf,
                   original_data.getFileSize()),
            0);
}

uintptr_t hex_to_uintptr(const std::string& hex_str) {
  std::string hex_str_no_prefix = hex_str.substr(2);
  uintptr_t result;
  std::stringstream ss;
  ss << std::hex << hex_str_no_prefix;
  ss >> result;
  return result;
}

class NvmePluginTest : public ::testing::Test {
 public:
  static void TearDownTestSuite() {
    auto& sm = StorageManager::getInstance();
    sm.unloadAll();
  }

 protected:
  void SetUp() override {}
  void TearDown() override {}

  static RelBuilderFactory getRelBuilderFactory() {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    return RelBuilderFactory{std::string(test_info->test_suite_name()) + "_" +
                             test_info->name()};
  }

  /**
   * @brief validate the result of a scan operation where there is only a
   * single partition per column. Assumes single threaded execution so that the
   * page_ids in the result are the same order as emitted by the scan
   */
  static void validate_page_id_result_1_part_per_col(const QueryResult& res) {
    std::stringstream res_str;
    res_str << res;
    std::string out_tuple;
    int tuple_count = 0;
    while (std::getline(res_str, out_tuple)) {
      std::istringstream iss(out_tuple);
      std::string page_id_as_ptr;
      int attr = 0;
      while (std::getline(iss, page_id_as_ptr, ',')) {
        auto page_id_as_uintptr = hex_to_uintptr(page_id_as_ptr);
        auto page_id = NvmePlugin::PageId_t::from_ptr(
            reinterpret_cast<void*>(page_id_as_uintptr));
        ASSERT_TRUE(NvmePlugin::PageId_t::isPageIdPtr(page_id_as_uintptr));
        ASSERT_EQ(page_id.getAttributeNo(), attr)
            << "tuple#: " << tuple_count << " " << page_id;
        ASSERT_EQ(page_id.getPartitionNo(), 0)
            << "tuple#: " << tuple_count << " " << page_id;
        ASSERT_EQ(page_id.getBlockNo(), tuple_count)
            << "tuple#: " << tuple_count << " " << page_id;

        attr += 1;
      }
      tuple_count += 1;
    }
  }

  /**
   * @brief validate the result of a scan operation where there are n partition
   * per column. Assumes partitions are all equally sized.  Assumes single
   * threaded execution so that the page_ids in the result are the same order as
   * emitted by the scan
   */
  static void validate_page_id_result_n_part_per_col(const QueryResult& res,
                                                     size_t n_parts_per_col) {
    std::stringstream res_str;
    res_str << res;
    std::string out_tuple;
    int tuple_count = 0;
    int expected_partition = 0;
    // all partitions have block_no starting from 0
    int expected_block_no = 0;
    while (std::getline(res_str, out_tuple)) {
      std::istringstream iss(out_tuple);
      std::string page_id_as_ptr;
      int attr = 0;
      while (std::getline(iss, page_id_as_ptr, ',')) {
        auto page_id = NvmePlugin::PageId_t::from_ptr(
            reinterpret_cast<void*>(hex_to_uintptr(page_id_as_ptr)));
        ASSERT_EQ(page_id.getAttributeNo(), attr)
            << "tuple#: " << tuple_count << " " << page_id;
        ASSERT_EQ(page_id.getPartitionNo(), expected_partition)
            << "tuple#: " << tuple_count << " " << page_id;
        ASSERT_EQ(page_id.getBlockNo(), expected_block_no)
            << "tuple#: " << tuple_count << " " << page_id;
        attr += 1;
      }
      tuple_count += 1;
      expected_partition += 1;
      if (expected_partition == n_parts_per_col) {
        expected_partition = 0;
        expected_block_no += 1;
      }
    }
  }

  static std::tuple<int32_t, int32_t, int32_t> calculate_min_max_count_int32(
      const std::filesystem::path& data_file) {
    std::ifstream file(data_file, std::ios::binary);
    int32_t max_value = std::numeric_limits<int32_t>::min();
    int32_t min_value = std::numeric_limits<int32_t>::max();
    int32_t count = 0;
    while (file) {
      int32_t value;
      file.read(reinterpret_cast<char*>(&value), sizeof(value));
      if (file.gcount() != sizeof(value)) {
        break;
      }
      count += 1;
      if (value > max_value) {
        max_value = value;
      }
      if (value < min_value) {
        min_value = value;
      }
    }
    return {min_value, max_value, count};
  }

  static std::vector<int32_t> splitStringToInt32(const std::string& str) {
    std::vector<int32_t> result;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
      result.push_back(std::stoi(token));
    }

    return result;
  }
};

TEST_F(NvmePluginTest, scan_one_col_one_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();

  SCOPED_TRACE("");
  validate_page_id_result_1_part_per_col(res);
}

TEST_F(NvmePluginTest, scan_two_col_one_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();

  SCOPED_TRACE("");
  validate_page_id_result_1_part_per_col(res);
}

TEST_F(NvmePluginTest, scan_one_col_two_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path, md_path};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();

  SCOPED_TRACE("");
  validate_page_id_result_n_part_per_col(res, 2);
}

TEST_F(NvmePluginTest, getPageIoInfoUsingMetadata) {
  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path, md_path};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);
  auto context = OlapParallelContext("test");
  NvmePlugin testPlugin(&context, relation_md);

  // Assuming the second block of the first attribute for testing
  NvmePlugin::PageId_t testPageId(0,   // CPU NUMA affinity,
                                  0,   // attribute number,
                                  0,   // partition number,
                                  1);  //  block number

  // Retrieve IO information for the test page
  auto page_io_info = testPlugin.getPageIoInfo(testPageId);

  // Load expected values from the metadata file
  NvmePlugin::AttributePartMetaData partMetaData(md_path);
  const uint64_t expected_offset = partMetaData.block_offsets[1];
  const auto expected_size = static_cast<size_t>(partMetaData.block_sizes[1]);

  // Can't test FD easily without a real file, but we can test the offset and
  // expected size
  EXPECT_EQ(*page_io_info.offset, expected_offset);
  EXPECT_EQ(*page_io_info.size, expected_size);
  EXPECT_EQ(page_io_info.is_compressed, false);
}

TEST_F(NvmePluginTest, scan_and_move_one_col_one_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(32, DeviceType::CPU)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "min"),
                               arg["c_custkey"].as("tmp", "max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 3);
  auto [expected_min, expected_max, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  EXPECT_EQ(split_output[0], expected_min);
  EXPECT_EQ(split_output[1], expected_max);
  EXPECT_EQ(split_output[2], expected_count);
}

TEST_F(NvmePluginTest, scan_and_move_one_col_two_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type = rel("customer.csv")(Int("c_custkey"));

  const std::filesystem::path md_path =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";
  std::vector<std::filesystem::path> attr_md{md_path, md_path};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_custkey"], attr_md);

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(32, DeviceType::CPU)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "min"),
                               arg["c_custkey"].as("tmp", "max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 3)
      << "expected a single tuple with 3 aggregates";
  auto [expected_min, expected_max, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  EXPECT_EQ(split_output[0], expected_min);
  EXPECT_EQ(split_output[1], expected_max);
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[2], 2 * expected_count);
}

TEST_F(NvmePluginTest, scan_and_move_two_col_two_part) {
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(32, DeviceType::CPU)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key);
  EXPECT_EQ(split_output[1], expected_max_key);
  EXPECT_EQ(split_output[2], expected_min_phone);
  EXPECT_EQ(split_output[3], expected_max_phone);
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count);
}

TEST_F(NvmePluginTest, scan_and_move_then_router_two_col_two_part) {
  // here the memmove is single threaded and then the reduction in parallelized
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(32, DeviceType::CPU)
                 .router(DegreeOfParallelism(4), 4, RoutingPolicy::LOCAL,
                         DeviceType::CPU)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .router(DegreeOfParallelism{1}, 32, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key);
  EXPECT_EQ(split_output[1], expected_max_key);
  EXPECT_EQ(split_output[2], expected_min_phone);
  EXPECT_EQ(split_output[3], expected_max_phone);
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count);
}

TEST_F(NvmePluginTest, scan_and_router_then_move_two_col_two_part) {
  // here the memmove and the reduction are parallelized
  // This tests that the affinitizers can work with page_ids and that
  // router/memmove propagate the tuple/block count properly
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .router(DegreeOfParallelism(4), 4, RoutingPolicy::LOCAL,
                         DeviceType::CPU)
                 .memmove(32, DeviceType::CPU)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .router(DegreeOfParallelism{1}, 32, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key);
  EXPECT_EQ(split_output[1], expected_max_key);
  EXPECT_EQ(split_output[2], expected_min_phone);
  EXPECT_EQ(split_output[3], expected_max_phone);
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count);
}

TEST_F(NvmePluginTest, scan_and_membrdcst_two_col_two_part) {
  GTEST_SKIP_("membrdcst is not yet supported");
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code"
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .router(DegreeOfParallelism(4), 4, RoutingPolicy::LOCAL,
                         DeviceType::CPU)
                 .membrdcst(DegreeOfParallelism(2), true)
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .router(DegreeOfParallelism{1}, 32, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key);
  EXPECT_EQ(split_output[1], expected_max_key);
  EXPECT_EQ(split_output[2], expected_min_phone);
  EXPECT_EQ(split_output[3], expected_max_phone);
  // 2 partitions that are the same base data, which are then broadcasted to 2
  // threads
  EXPECT_EQ(split_output[4], 4 * expected_count);
#pragma clang diagnostic pop
}

TEST_F(NvmePluginTest, gpu_scan_and_move_two_col_two_part) {
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() == 0) {
    GTEST_SKIP_("No GPUs available");
  }
  set_exec_location_on_scope exec(topo.getGpus()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(4, DeviceType::GPU)
                 .to_gpu()
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .to_cpu()
                 .router(DegreeOfParallelism{1}, 4, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key)
      << "full output " << output.str();
  EXPECT_EQ(split_output[1], expected_max_key)
      << "full output " << output.str();
  EXPECT_EQ(split_output[2], expected_min_phone)
      << "full output " << output.str();
  EXPECT_EQ(split_output[3], expected_max_phone)
      << "full output " << output.str();
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count)
      << "full output " << output.str();
}

TEST_F(NvmePluginTest, gpu_scan_and_move_two_col_four_part_large) {
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() == 0) {
    GTEST_SKIP_("No GPUs available");
  }
  set_exec_location_on_scope exec(topo.getGpus()[0]);

  RecordType my_record_type =
      rel("lineorder.csv")(Int("lo_partkey"), Int("lo_suppkey"));

  const std::vector<std::filesystem::path> partkey_md_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/"
      "lineorder.csv.lo_partkey_0_12.metadata.json",
      "/nvme22/nicholso/data/sbm1000_1_12/"
      "lineorder.csv.lo_partkey_1_12.metadata.json",
      "/nvme29/nicholso/data/sbm1000_2_12/"
      "lineorder.csv.lo_partkey_2_12.metadata.json",
      "/nvme24/nicholso/data/sbm1000_3_12/"
      "lineorder.csv.lo_partkey_3_12.metadata.json"};

  const std::vector<std::filesystem::path> partkey_data_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/lineorder.csv.lo_partkey_0_12",
      "/nvme22/nicholso/data/sbm1000_1_12/lineorder.csv.lo_partkey_1_12",
      "/nvme29/nicholso/data/sbm1000_2_12/lineorder.csv.lo_partkey_2_12",
      "/nvme24/nicholso/data/sbm1000_3_12/lineorder.csv.lo_partkey_3_12"};

  const std::vector<std::filesystem::path> suppkey_md_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/"
      "lineorder.csv.lo_suppkey_0_12.metadata.json",
      "/nvme22/nicholso/data/sbm1000_1_12/"
      "lineorder.csv.lo_suppkey_1_12.metadata.json",
      "/nvme29/nicholso/data/sbm1000_2_12/"
      "lineorder.csv.lo_suppkey_2_12.metadata.json",
      "/nvme24/nicholso/data/sbm1000_3_12/"
      "lineorder.csv.lo_suppkey_3_12.metadata.json"};

  const std::vector<std::filesystem::path> suppkey_data_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/lineorder.csv.lo_suppkey_0_12",
      "/nvme22/nicholso/data/sbm1000_1_12/lineorder.csv.lo_suppkey_1_12",
      "/nvme29/nicholso/data/sbm1000_2_12/lineorder.csv.lo_suppkey_2_12",
      "/nvme24/nicholso/data/sbm1000_3_12/lineorder.csv.lo_suppkey_3_12"};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["lo_partkey"],
                           partkey_md_paths);

  relation_md.emplace_back(meta_data_records_map["lo_suppkey"],
                           suppkey_md_paths);

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res =
      factory.getBuilder()
          .scan(relation_md)
          .memmove(4, DeviceType::GPU)
          .to_gpu()
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["lo_partkey"].as("tmp", "partkey_min"),
                        arg["lo_partkey"].as("tmp", "partkey_max"),
                        arg["lo_suppkey"].as("tmp", "suppkey_min"),
                        arg["lo_suppkey"].as("tmp", "suppkey_max"),
                        expression_t{1}.as("tmp", "count")};
              },
              {MIN, MAX, MIN, MAX, SUM})
          .to_cpu()
          .router(DegreeOfParallelism{1}, 4, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["partkey_min"], arg["partkey_max"],
                        arg["suppkey_min"], arg["suppkey_max"], arg["count"]};
              },
              {MIN, MAX, MIN, MAX, SUM})
          .print(pg("pm-csv"))
          .prepare()
          .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_partkey, expected_max_partkey, expected_count,
        expected_min_suppkey, expected_max_suppkey] = [&]() {
    auto& tp = ThreadPool::getInstance();
    std::vector<std::future<std::tuple<int32_t, int32_t, int32_t>>>
        partkey_futures;
    std::vector<std::future<std::tuple<int32_t, int32_t, int32_t>>>
        suppkey_futures;

    for (const auto& pk : partkey_data_paths) {
      partkey_futures.emplace_back(
          tp.enqueue(calculate_min_max_count_int32, pk));
    }

    for (const auto& sk : suppkey_data_paths) {
      suppkey_futures.emplace_back(
          tp.enqueue(calculate_min_max_count_int32, sk));
    }

    int32_t min_partkey = std::numeric_limits<int32_t>::max();
    int32_t max_partkey = std::numeric_limits<int32_t>::min();
    int32_t count = 0;
    for (auto& f : partkey_futures) {
      auto [min, max, c] = f.get();
      min_partkey = std::min(min_partkey, min);
      max_partkey = std::max(max_partkey, max);
      count += c;
    }
    int32_t min_suppkey = std::numeric_limits<int32_t>::max();
    int32_t max_suppkey = std::numeric_limits<int32_t>::min();
    for (auto& f : suppkey_futures) {
      auto [min, max, _] = f.get();
      min_suppkey = std::min(min_suppkey, min);
      max_suppkey = std::max(max_suppkey, max);
    }
    return std::make_tuple(min_partkey, max_partkey, count, min_suppkey,
                           max_suppkey);
  }();

  EXPECT_EQ(split_output[0], expected_min_partkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[1], expected_max_partkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[2], expected_min_suppkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[3], expected_max_suppkey)
      << "full output " << output.str();
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], expected_count) << "full output " << output.str();
}

TEST_F(NvmePluginTest,
       gpu_scan_and_move_two_col_four_part_large_with_parallel_load) {
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() == 0) {
    GTEST_SKIP_("No GPUs available");
  }
  set_exec_location_on_scope exec(topo.getGpus()[0]);

  RecordType my_record_type =
      rel("lineorder.csv")(Int("lo_partkey"), Int("lo_suppkey"));

  const std::vector<std::filesystem::path> partkey_md_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/"
      "lineorder.csv.lo_partkey_0_12.metadata.json",
      "/nvme22/nicholso/data/sbm1000_1_12/"
      "lineorder.csv.lo_partkey_1_12.metadata.json",
      "/nvme29/nicholso/data/sbm1000_2_12/"
      "lineorder.csv.lo_partkey_2_12.metadata.json",
      "/nvme24/nicholso/data/sbm1000_3_12/"
      "lineorder.csv.lo_partkey_3_12.metadata.json"};

  const std::vector<std::filesystem::path> partkey_data_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/lineorder.csv.lo_partkey_0_12",
      "/nvme22/nicholso/data/sbm1000_1_12/lineorder.csv.lo_partkey_1_12",
      "/nvme29/nicholso/data/sbm1000_2_12/lineorder.csv.lo_partkey_2_12",
      "/nvme24/nicholso/data/sbm1000_3_12/lineorder.csv.lo_partkey_3_12"};

  const std::vector<std::filesystem::path> suppkey_md_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/"
      "lineorder.csv.lo_suppkey_0_12.metadata.json",
      "/nvme22/nicholso/data/sbm1000_1_12/"
      "lineorder.csv.lo_suppkey_1_12.metadata.json",
      "/nvme29/nicholso/data/sbm1000_2_12/"
      "lineorder.csv.lo_suppkey_2_12.metadata.json",
      "/nvme24/nicholso/data/sbm1000_3_12/"
      "lineorder.csv.lo_suppkey_3_12.metadata.json"};

  const std::vector<std::filesystem::path> suppkey_data_paths = {
      "/nvme16/nicholso/data/sbm1000_0_12/lineorder.csv.lo_suppkey_0_12",
      "/nvme22/nicholso/data/sbm1000_1_12/lineorder.csv.lo_suppkey_1_12",
      "/nvme29/nicholso/data/sbm1000_2_12/lineorder.csv.lo_suppkey_2_12",
      "/nvme24/nicholso/data/sbm1000_3_12/lineorder.csv.lo_suppkey_3_12"};

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["lo_partkey"],
                           partkey_md_paths);

  relation_md.emplace_back(meta_data_records_map["lo_suppkey"],
                           suppkey_md_paths);

  RelBuilderFactory factory = getRelBuilderFactory();
  auto res =
      factory.getBuilder()
          .scan(relation_md)
          .router(
              DegreeOfParallelism(topology::getInstance().getGpuCount() * 12),
              2, RoutingPolicy::LOCAL, DeviceType::CPU,
              std::make_unique<GPUAffinitizer>())
          .memmove(4, DeviceType::GPU)
          .router(DegreeOfParallelism{topology::getInstance().getGpuCount()}, 8,
                  RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<GPUAffinitizer>())
          .to_gpu()
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["lo_partkey"].as("tmp", "partkey_min"),
                        arg["lo_partkey"].as("tmp", "partkey_max"),
                        arg["lo_suppkey"].as("tmp", "suppkey_min"),
                        arg["lo_suppkey"].as("tmp", "suppkey_max"),
                        expression_t{1}.as("tmp", "count")};
              },
              {MIN, MAX, MIN, MAX, SUM})
          .to_cpu()
          .router(DegreeOfParallelism{1}, 4, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["partkey_min"], arg["partkey_max"],
                        arg["suppkey_min"], arg["suppkey_max"], arg["count"]};
              },
              {MIN, MAX, MIN, MAX, SUM})
          .print(pg("pm-csv"))
          .prepare()
          .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_partkey, expected_max_partkey, expected_count,
        expected_min_suppkey, expected_max_suppkey] = [&]() {
    auto& tp = ThreadPool::getInstance();
    std::vector<std::future<std::tuple<int32_t, int32_t, int32_t>>>
        partkey_futures;
    std::vector<std::future<std::tuple<int32_t, int32_t, int32_t>>>
        suppkey_futures;

    for (const auto& pk : partkey_data_paths) {
      partkey_futures.emplace_back(
          tp.enqueue(calculate_min_max_count_int32, pk));
    }

    for (const auto& sk : suppkey_data_paths) {
      suppkey_futures.emplace_back(
          tp.enqueue(calculate_min_max_count_int32, sk));
    }

    int32_t min_partkey = std::numeric_limits<int32_t>::max();
    int32_t max_partkey = std::numeric_limits<int32_t>::min();
    int32_t count = 0;
    for (auto& f : partkey_futures) {
      auto [min, max, c] = f.get();
      min_partkey = std::min(min_partkey, min);
      max_partkey = std::max(max_partkey, max);
      count += c;
    }
    int32_t min_suppkey = std::numeric_limits<int32_t>::max();
    int32_t max_suppkey = std::numeric_limits<int32_t>::min();
    for (auto& f : suppkey_futures) {
      auto [min, max, _] = f.get();
      min_suppkey = std::min(min_suppkey, min);
      max_suppkey = std::max(max_suppkey, max);
    }
    return std::make_tuple(min_partkey, max_partkey, count, min_suppkey,
                           max_suppkey);
  }();

  EXPECT_EQ(split_output[0], expected_min_partkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[1], expected_max_partkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[2], expected_min_suppkey)
      << "full output " << output.str();
  EXPECT_EQ(split_output[3], expected_max_suppkey)
      << "full output " << output.str();
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], expected_count) << "full output " << output.str();
}

TEST_F(NvmePluginTest, gpu_scan_and_move_two_col_two_part_staging) {
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() == 0) {
    GTEST_SKIP_("No GPUs available");
  }
  set_exec_location_on_scope exec(topo.getGpus()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  const std::filesystem::path md_path_c_phone =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_phone.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "inputs/nvme-plugin-tests/ssb100_customer.csv.c_custkey.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(4, DeviceType::GPU,
                          std::make_optional<std::vector<bool>>(
                              {true, false}))  /// staging 2nd col
                 .to_gpu()
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .to_cpu()
                 .router(DegreeOfParallelism{1}, 4, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  auto split_output = splitStringToInt32(output.str());
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates";
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm100/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key)
      << "full output " << output.str();
  EXPECT_EQ(split_output[1], expected_max_key)
      << "full output " << output.str();
  EXPECT_EQ(split_output[2], expected_min_phone)
      << "full output " << output.str();
  EXPECT_EQ(split_output[3], expected_max_phone)
      << "full output " << output.str();
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count)
      << "full output " << output.str();
}

TEST_F(NvmePluginTest, gpu_scan_and_move_two_col_two_part_compressed) {
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() == 0) {
    GTEST_SKIP_("No GPUs available");
  }
  set_exec_location_on_scope exec(topo.getGpus()[0]);

  RecordType my_record_type =
      rel("customer.csv")(Int("c_phone"), Int("c_custkey"));

  // TODO fix server specific paths
  const std::filesystem::path md_path_c_phone =
      "/nvme14/nicholso/data/compressed_ssbm1000/"
      "customer.csv.c_phone_0_1.metadata.json";
  const std::filesystem::path md_path_c_custkey =
      "/nvme14/nicholso/data/compressed_ssbm1000/"
      "customer.csv.c_custkey_0_1.metadata.json";

  auto meta_data_records_map = my_record_type.getArgsMap();
  std::vector<std::pair<RecordAttribute*, std::vector<std::filesystem::path>>>
      relation_md;
  relation_md.emplace_back(meta_data_records_map["c_phone"],
                           std::vector{md_path_c_phone, md_path_c_phone});

  relation_md.emplace_back(meta_data_records_map["c_custkey"],
                           std::vector{md_path_c_custkey, md_path_c_custkey});

  RelBuilderFactory factory = getRelBuilderFactory();
  //  res should hold the page_ids in csv format for the 6 blocks
  auto res = factory.getBuilder()
                 .scan(relation_md)
                 .memmove(4, DeviceType::GPU)
                 .to_gpu()
                 .unpack()
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["c_custkey"].as("tmp", "key_min"),
                               arg["c_custkey"].as("tmp", "key_max"),
                               arg["c_phone"].as("tmp", "phone_min"),
                               arg["c_phone"].as("tmp", "phone_max"),
                               expression_t{1}.as("tmp", "count")};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .to_cpu()
                 .router(DegreeOfParallelism{1}, 4, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto& arg) -> std::vector<expression_t> {
                       return {arg["key_min"], arg["key_max"], arg["phone_min"],
                               arg["phone_max"], arg["count"]};
                     },
                     {MIN, MAX, MIN, MAX, SUM})
                 .print(pg("pm-csv"))
                 .prepare()
                 .execute();
  std::stringstream output;
  output << res;
  std::string res_string = output.str();
  auto split_output = splitStringToInt32(res_string);
  EXPECT_EQ(split_output.size(), 5)
      << "expected a single tuple with 5 aggregates instead of " << res_string;
  auto [expected_min_key, expected_max_key, expected_count] =
      calculate_min_max_count_int32("inputs/ssbm1000/customer.csv.c_custkey");
  auto [expected_min_phone, expected_max_phone, _] =
      calculate_min_max_count_int32("inputs/ssbm1000/customer.csv.c_phone");
  EXPECT_EQ(split_output[0], expected_min_key);
  EXPECT_EQ(split_output[1], expected_max_key);
  EXPECT_EQ(split_output[2], expected_min_phone);
  EXPECT_EQ(split_output[3], expected_max_phone);
  // 2 partitions that are the same base data
  EXPECT_EQ(split_output[4], 2 * expected_count);
}

#pragma clang diagnostic pop
