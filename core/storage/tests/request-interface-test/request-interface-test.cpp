/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <storage/storage-manager.hpp>

class RequestInterfaceTest : public ::testing::Test {};

using namespace std::chrono_literals;

class FakeFileLoadCounter {
 protected:
  std::atomic<int> cnt = 0;
  static constexpr auto fileName = "non-existent-file";

  virtual FileRecord callback() {
    ++cnt;
    return FileRecord({});
  }

 public:
  FakeFileLoadCounter() {
    LOG(INFO) << "test2";
    StorageManager::getInstance().setLoader(
        fileName,
        [&](StorageManager &, const std::string &,
            size_t typeSize) -> FileRecord { return callback(); },
        true);
  }
  FakeFileLoadCounter(const FakeFileLoadCounter &) = delete;
  FakeFileLoadCounter(FakeFileLoadCounter &&) = delete;
  FakeFileLoadCounter &operator=(const FakeFileLoadCounter &) const = delete;
  FakeFileLoadCounter &operator=(FakeFileLoadCounter &&) const = delete;
  virtual ~FakeFileLoadCounter() = default;

  auto file() const { return fileName; }

  explicit(false) operator size_t() const { return cnt; }
};

class FakeFileLoadCounterDelayed : public FakeFileLoadCounter {
 public:
  static constexpr std::chrono::milliseconds sleep_duration{500};

 protected:
  FileRecord callback() override {
    std::this_thread::sleep_for(sleep_duration);
    ++cnt;
    return FileRecord({});
  }

 public:
  using FakeFileLoadCounter::FakeFileLoadCounter;
};

TEST_F(RequestInterfaceTest, request_but_do_not_load) {
  FakeFileLoadCounter cnt{};
  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);
  EXPECT_EQ(cnt, 0);
}

TEST_F(RequestInterfaceTest, pin_unpin_max_one_load) {
  FakeFileLoadCounter cnt{};
  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);
  EXPECT_EQ(cnt, 0);

  rq.pin();

  // To check if the loader should be forced to be called here
  // Thought: pin promises that the file should stay in memory,
  //   but until the file is really requested, there is no reason
  //   we should force the loading to happen immediately.
  //   For example, we may know that when file X.txt is loaded it
  //   should stay in memory until it's unpinned, but we may be
  //   better off to leavy the storage manager to return the file
  //   only after getSegments() (of some FileRequest for the same file)
  //   has returned so that the storage manager is allowed to delay
  //   the loading of files (e.g., a query if finishing soon)
  // EXPECT_EQ(cnt, 1);

  rq.unpin();
  EXPECT_LE(cnt, 1);
}

TEST_F(RequestInterfaceTest, do_pin_load_once_unpin) {
  FakeFileLoadCounter cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.pin();
  // Requests the segments so that the storage manager loads them
  ((void)rq.getSegments());

  EXPECT_EQ(cnt, 1);

  rq.unpin();
  EXPECT_EQ(cnt, 1);
}

TEST_F(RequestInterfaceTest, intent_loads_max_one) {
  FakeFileLoadCounter cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);
  EXPECT_EQ(cnt, 0);

  rq.registerIntent();
  EXPECT_LE(cnt, 1);
}

TEST_F(RequestInterfaceTest, do_pin_loadonce_unpin) {
  FakeFileLoadCounter cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();

  rq.pin();
  // Requests the segments so that the storage manager loads them
  ((void)rq.getSegments());

  EXPECT_EQ(cnt, 1);

  auto rq2 =
      StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq2.pin();
  // If the file is loaded, we should not reload it, while pinned
  EXPECT_EQ(cnt, 1);
  rq2.unpin();

  rq.unpin();
  EXPECT_EQ(cnt, 1);
}

TEST_F(RequestInterfaceTest, intent_while_pinned_should_noop) {
  FakeFileLoadCounter cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();

  rq.pin();
  // Requests the segments so that the storage manager loads them
  ((void)rq.getSegments());

  EXPECT_EQ(cnt, 1);

  {
    auto rq2 =
        StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);
    rq2.registerIntent();
  }

  rq.unpin();
  EXPECT_EQ(cnt, 1);
}

TEST_F(RequestInterfaceTest, get_segments_waits_for_loader) {
  FakeFileLoadCounterDelayed cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();

  rq.pin();
  // Requests the segments so that the storage manager loads them
  ((void)rq.getSegments());
  EXPECT_EQ(cnt, 1);

  rq.unpin();
  EXPECT_EQ(cnt, 1);
}

TEST_F(RequestInterfaceTest, get_segments_waits_for_loader_pinunipinloops) {
  FakeFileLoadCounterDelayed cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();

  for (size_t i = 0; i < 10; ++i) {
    rq.pin();
    // Requests the segments so that the storage manager loads them
    ((void)rq.getSegments());
    EXPECT_LE(cnt, i + 1);  // LE instead of EQ as pin does not force unload

    rq.unpin();
    EXPECT_LE(cnt, i + 1);  // LE instead of EQ as pin does not force unload
  }
}

TEST_F(RequestInterfaceTest, get_segments_w_count_before_pin) {
  FakeFileLoadCounterDelayed cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();
  EXPECT_LE(cnt, 1);

  ((void)rq.getSegmentCount());
  EXPECT_LE(cnt, 2);

  {
    rq.pin();
    // Requests the segments so that the storage manager loads them
    ((void)rq.getSegments());
    EXPECT_LE(cnt, 3);  // LE instead of EQ as pin does not force unload

    rq.unpin();
    EXPECT_LE(cnt, 3);  // LE instead of EQ as pin does not force unload
  }
}

TEST_F(RequestInterfaceTest, async_read_by_intent) {
  FakeFileLoadCounterDelayed cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();
  EXPECT_EQ(cnt, 0);

  std::this_thread::sleep_for(3 * cnt.sleep_duration);
  EXPECT_LE(cnt, 1);
}

TEST_F(RequestInterfaceTest,
       async_read_by_concurrent_intent_reads_at_most_one) {
  FakeFileLoadCounterDelayed cnt{};
  EXPECT_EQ(cnt, 0);

  auto rq = StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);

  rq.registerIntent();
  EXPECT_EQ(cnt, 0);
  auto rq2 =
      StorageManager::getInstance().request(cnt.file(), 4, FROM_REGISTRY);
  rq2.registerIntent();

  // Take the count here, just in case the first one finished too
  // close/before registering the second intent
  size_t cntAfterSecondIntent = cnt;

  std::this_thread::sleep_for(4 * cnt.sleep_duration);
  EXPECT_LE(cnt, cntAfterSecondIntent + 1);
}
