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

class cow_arenas_t {
  static int shm_fd;
  static size_t size_bytes;
  static int8_t *save_to;
  static int8_t *start;

 public:
  static void handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    //   printf("\nAttempt to access memory at address %p\n", siginfo->si_addr);
    // #ifdef LINUX_64BIT
    //   printf("Instruction pointer: %p\n",
    //          (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
    // #elif defined(LINUX_32BIT)
    //   printf("Instruction pointer: %p\n",
    //          (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
    // #endif
    void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));
    auto offset = ((int8_t *)addr) - start;

    // auto f = mremap(src2 + offset, page_size, page_size,
    //                 MREMAP_FIXED | MREMAP_MAYMOVE, addr);
    // assert(f != MAP_FAILED);
    // assert(f == addr);
    std::memcpy(save_to + offset, addr, page_size);

    mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    // signal(sig, SIG_DFL);
    // raise(sig);
#pragma clang diagnostic pop
  }

  static int *oltp_arena;
  int *olap_arena;

  static void init(size_t size_bytes) {
    cow_arenas_t::size_bytes = size_bytes;
    shm_fd = memfd_create("cow", 0);
    ASSERT_GE(shm_fd, 0);
    ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
    size_t N = size_bytes / sizeof(int);

    // Mark MAP_SHARED so that we can map the same memory multiple times
    oltp_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                             MAP_SHARED, shm_fd, 0);
    ASSERT_NE(oltp_arena, MAP_FAILED);
    start = (int8_t *)oltp_arena;

    // Put some data
    for (size_t i = 0; i < N; ++i) {
      oltp_arena[i] = i;
    }

    EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
              (int)(N * (N - 1) / 2));

    // Create snapshot

    // 0. Install handler to lazily steal and save data to-be-overwritten by
    // OLTP
    struct sigaction act {};
    sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    act.sa_sigaction = cow_arenas_t::handler;
#pragma clang diagnostic pop
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &act, nullptr);
  }

  int *oltp() { return oltp_arena; }
  int *olap() { return olap_arena; }

  void create_snapshot(void *place_at = nullptr) {
    // Should we mprotect before or after?
    // ANSWER:
    //    If we mprotect after, we may loose some updates!
    //    If we mprotect before, our handler may not know where to copy some
    //    updates!

    // 1. Mark OLTP side as RO
    assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

    // 2. Create a second mapping of the OLTP data for the OLAP side
    // 2.i   The mapping can not be MAP_SHARED as we do not want to create
    // changes private to the OLTP side.
    // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
    // side, we have to catch OLTP writes, still the underlying data and
    // replicate the, to the OLAP side.
    olap_arena =
        (int *)mmap(place_at, size_bytes, PROT_WRITE | PROT_READ,
                    MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
    assert(olap_arena != MAP_FAILED);
    save_to = (int8_t *)olap_arena;
    // return olap_arena;
  }

  void destroy_snapshot() {
    // time_block t{"T2: "};
    // Throw away snapshot
    munmap(olap_arena, size_bytes);
  }
};

TEST_F(SnapshottingTest, cow2) {
  std::string testLabel = "cow2";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  cow_arenas_t::init(size_bytes);

  size_t N = size_bytes / sizeof(int);

  cow_arenas_t arenas{};
  arenas.create_snapshot();
  // // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

  int *olap_arena = arenas.olap();
  int *oltp_arena = arenas.oltp();
  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 1;
  }

  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
            (int)(N * (N - 1) / 2));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)N);

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    ++(oltp_arena[i]);
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(2 * N));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 5;
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(2 * N));
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(5 * N));

  // benchmark_writeall_sequencial(oltp_arena, N, create_snapshot,
  //                               destroy_snapshot);
};

int cow_arenas_t::shm_fd;
size_t cow_arenas_t::size_bytes;
int8_t *cow_arenas_t::save_to;
int8_t *cow_arenas_t::start;
int *cow_arenas_t::oltp_arena;

