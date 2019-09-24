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

#include <arpa/inet.h>
#include <err.h>
#include <gflags/gflags.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <common/olap-common.hpp>
#include <cstring>
#include <network/infiniband/infiniband-manager.hpp>
#include <util/timing.hpp>

#include "cli-flags.hpp"
#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "plan/prepared-statement.hpp"
#include "storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/profiling.hpp"

extern bool print_generated_code;

// https://stackoverflow.com/a/25829178/1237824
std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) return str;
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

// https://stackoverflow.com/a/7756105/1237824
bool starts_with(const std::string &s1, const std::string &s2) {
  return s2.size() <= s1.size() && s1.compare(0, s2.size(), s2) == 0;
}

constexpr size_t clen(const char *str) {
  return (*str == 0) ? 0 : clen(str + 1) + 1;
}

const char *catalogJSON = "inputs";

auto executePlan(const char *label, const char *planPath,
                 const char *catalogJSON) {
  auto prepared = PreparedStatement::from(planPath, label, catalogJSON);
  for (size_t i = 1; i < FLAGS_repeat; ++i) prepared.execute();
  return prepared.execute();
}

auto executePlan(const char *label, const char *planPath) {
  return executePlan(label, planPath, catalogJSON);
}

class unlink_upon_exit {
  size_t query;
  std::string label_prefix;

  std::string last_label;

  std::unique_ptr<QueryResult> last_result;

 public:
  unlink_upon_exit()
      : query(0),
        label_prefix("raw_server_" + std::to_string(getpid()) + "_q"),
        last_label("") {}

  unlink_upon_exit(size_t unique_id)
      : query(0),
        label_prefix("raw_server_" + std::to_string(unique_id) + "_q"),
        last_label("") {}

  std::string get_label() const { return last_label; }

  std::string inc_label() {
    last_label = label_prefix + std::to_string(query++);
    return last_label;
  }

  void store(QueryResult &&qr) {
    last_result = std::make_unique<QueryResult>(std::move(qr));
  }
};

std::string runPlanFile(std::string plan, unlink_upon_exit &uue,
                        bool echo = true) {
  std::string label = uue.inc_label();
  auto qr = executePlan(label.c_str(), plan.c_str());

  if (echo) {
    std::cout << "result echo" << std::endl;
    std::cout << qr << std::endl;
  }

  uue.store(std::move(qr));

  return label;
}

/**
 * Protocol:
 *
 * Communication is done over stdin/stdout
 * Command spans at most one line
 * Every line either starts with a command keyword or it should be IGNORED and
 *      considered a comment
 * Input commands:
 *
 *      quit
 *          Kills the raw-jit-executor engine
 *
 *      execute plan <plan_description>
 *          Executes the plan described from the <plan_description>
 *          It will either result in an error command send back, or a result one
 *
 *          Valid plan descriptions:
 *
 *              from file <file_path>
 *                  Reads the plan from the file pointed by the <file_path>
 *                  The file path is either an absolute path, or a path relative
 *                  to the current working directory
 *
 *     echo <object_to_echo>
 *          Switched on/off the echoing of types of results. When switched on,
 *          in general, replies with the specific type of object that were
 *          to be written in files, are also echoed to stdout
 *
 *          Valid to-echo-objects:
 *              results (on/off)
 *                  Prints results in output as well.
 *                  Use with causion! Results may be binary or contain new lines
 *                  with keywords!
 *                  Default: off
 *
 * Output commands:
 *      ready
 *          Send to the client when the raw-jit-executor is ready to start
 *          receiving commands
 *      error [(<reason>)]
 *          Specifies that a previous command or the engine failed.
 *          The optional (<reason>) specified in parenthesis a human-readable
 *          explanation of the error. The error may be fatal or not.
 *      result <result_description>
 *          Specifies the result of the previous command, if any
 *
 *          Valid result descriptions:
 *              in file <file_path>
 *                  The result is saved in file pointed by the <file_path>
 *                  The file path is either an absolute path, or a path relative
 *                  to the current working directory
 *              echo
 *                  The following line/lines are results printed into stdout
 */
int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap(
      "Simple command line interface for proteus", &argc, &argv);

  bool echo = false;

  set_exec_location_on_scope affg{topology::getInstance().getGpus()[1]};
  set_exec_location_on_scope aff{topology::getInstance().getCpuNumaNodes()[1]};

  if (FLAGS_primary || FLAGS_secondary) {
    assert(FLAGS_port <= std::numeric_limits<uint16_t>::max());
    InfiniBandManager::init(FLAGS_url, static_cast<uint16_t>(FLAGS_port),
                            FLAGS_primary, FLAGS_ipv4);

    std::cout << "AsdADA" << std::endl;

    void *ptr = BlockManager::get_buffer();
    ((int *)ptr)[0] = 42 + FLAGS_primary;

    auto &sub = InfiniBandManager::subscribe();
    std::cout << "sub" << std::endl;

    for (int i = 0; i < 1; ++i) {
      if (FLAGS_primary) {
        std::cout << "send" << std::endl;
        void *ptr = BlockManager::get_buffer();
        ((int *)ptr)[0] = 45;
        InfiniBandManager::send(ptr, 4);
        std::cout << "send done" << std::endl;
      } else {
        sleep(2);
        std::cout << "wait" << std::endl;
        sub.wait();
        std::cout << "wait done" << std::endl;
        //      auto v = sub.wait();
        //      BlockManager::release_buffer((int32_t *) v.data);
      }

      if (FLAGS_primary) {
        sub.wait();
        //      auto v = sub.wait();
        //      BlockManager::release_buffer((int32_t *) v.data);
      } else {
        std::cout << "send" << std::endl;
        void *ptr = BlockManager::get_buffer();
        ((int *)ptr)[0] = 44;
        InfiniBandManager::send(ptr, 4);
        std::cout << "send done" << std::endl;
      }
    }

    constexpr size_t rlimit = 0x1000000000000;
    if (FLAGS_primary) {
      auto x = sub.wait();
      BlockManager::release_buffer((int32_t *)x.data);
      auto f = BlockManager::get_buffer();
      int32_t j = 0;
      for (size_t s : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        time_block t{"Tg:" + std::to_string(s) + ": "};
        int32_t sum = 0;

        nvtxRangePushA("reading");
        do {
          nvtxRangePushA("waiting");
          auto x = sub.wait();
          nvtxRangePop();
          if (x.size == 0) break;
          if (++j % rlimit == 0) {
            InfiniBandManager::send(f, 4);
            j = 0;
          }
          assert(x.size % 4 == 0);
          size_t size = x.size / 4;
          int32_t *data = (int32_t *)x.data;
          for (size_t i = 0; i < std::min(size_t{1}, size); ++i) {
            sum += data[i];
          }
          // MemoryManager::freePinned(sub.wait().data);
          BlockManager::release_buffer(data);
        } while (true);
        nvtxRangePop();
        std::cout << sum << std::endl;
        f[0] = sum;
        InfiniBandManager::send(f, 4);
        std::cout << sum << std::endl;
      }
    } else {
      auto v = StorageManager::getInstance()
                   .getOrLoadFile("inputs/ssbm100/lineorder.csv.lo_orderdate",
                                  4, PINNED)
                   .get();

      InfiniBandManager::reg((void *)v[0].data, v[0].size);
      profiling::resume();
      InfiniBandManager::send((char *)v[0].data, 0);
      int32_t j = 0;
      assert(v.size() == 1);
      for (size_t s : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        time_block t{"Tg:" + std::to_string(s) + ": "};

        {
          // constexpr size_t buff_size = ((size_t)16) * 1024 * 1024 * 1024;
          size_t buff_size = BlockManager::buffer_size / s;
          time_block t{"T: "};
          for (size_t i = 0; i < v[0].size; i += buff_size) {
            // LOG(INFO) << std::min(buff_size, v[0].size - i);
            InfiniBandManager::send(((char *)v[0].data) + i,
                                    std::min(buff_size, v[0].size - i));
            // std::cout << (i * 100 / v[0].size) << std::endl;
            if (++j % rlimit == 0) {
              BlockManager::release_buffer((int32_t *)sub.wait().data);
              j = 0;
            }
            // MemoryManager::freePinned(sub.wait().data);
            //        std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          // auto x = sub.wait();
          // assert(x.size == 4);
          // std::cout << ((int32_t *)x.data)[0] << std::endl;
        }
        InfiniBandManager::send((char *)v[0].data, 0);

        auto x = sub.wait();
        std::cout << ((int32_t *)x.data)[0] << std::endl;
        BlockManager::release_buffer((int32_t *)x.data);
      }
      profiling::pause();

      std::cout << "DONE" << std::endl;
      InfiniBandManager::disconnectAll();
      InfiniBandManager::unreg((void *)v[0].data);
      StorageManager::getInstance().unloadAll();
    }

    //    if (FLAGS_primary) {
    //      void * ptr = BlockManager::get_buffer();
    //      ((int *) ptr)[0] = 42;
    //      InfiniBandManager::send(ptr, 1024);
    //    } else {
    //      sub.wait();
    ////      auto v = sub.wait();
    ////      BlockManager::release_buffer((int32_t *) v.data);
    //      InfiniBandManager::disconnectAll();
    //    }

    //    if (FLAGS_primary) InfiniBandManager::disconnectAll();

    InfiniBandManager::deinit();
    return 0;
  }

  if (FLAGS_secondary) {
    struct sockaddr_in address;
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    std::string hello = "Hello from client";
    char buffer[1024] = {0};
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      printf("\n Socket creation error \n");
      return -1;
    }

    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(80818);

    struct addrinfo hints = {}, *addrs;
    char port_str[16] = {};
    int err = getaddrinfo(argv[2], "80818", &hints, &addrs);
    if (err != 0) {
      fprintf(stderr, "%s: %s\n", argv[2], gai_strerror(err));
      abort();
    }

    int sd;

    for (struct addrinfo *addr = addrs; addr != nullptr; addr = addr->ai_next) {
      sd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
      if (sd == -1) {
        err = errno;
        break;  // if using AF_UNSPEC above instead of AF_INET/6 specifically,
        // replace this 'break' with 'continue' instead, as the 'ai_family'
        // may be different on the next iteration...
      }

      if (connect(sd, addr->ai_addr, addr->ai_addrlen) == 0) break;

      err = errno;

      close(sd);
      sd = -1;
    }

    freeaddrinfo(addrs);

    if (sd == -1) {
      fprintf(stderr, "%s: %s\n", argv[2], strerror(err));
      abort();
    }

    sock = sd;
    //
    //    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
    //    0)
    //    {
    //      printf("\nConnection Failed \n");
    //      return -1;
    //    }
    send(sock, hello.c_str(), hello.size(), 0);
    printf("Hello message sent\n");
    valread = read(sock, buffer, 1024);
    printf("%s\n", buffer);
    return 0;
  }
  if (FLAGS_primary) {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    std::string hello = "Hello from server";
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
      std::cerr << "socket failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
      std::cerr << "setsockopt" << std::endl;
      exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(80818);

    // Forcefully attaching socket to the port 8080
    if (::bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
      std::cerr << "bind failed" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
      std::cerr << "listen" << std::endl;
      exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                             (socklen_t *)&addrlen)) < 0) {
      std::cerr << "accept" << std::endl;
      exit(EXIT_FAILURE);
    }
    printf("Hello messag2e sent\n");
    //    int n;
    //    while ( (n = read(new_socket, buffer, 1024-1)) > 0)
    //    {
    //      buffer[n] = 0;
    //      if(fputs(buffer, stdout) == EOF)
    //      {
    //        printf("\n Error : Fputs error\n");
    //      }
    //    }
    valread = read(new_socket, buffer, 1024);
    std::cout << valread << std::endl;
    printf("%s\n", buffer);
    send(new_socket, hello.c_str(), hello.size(), 0);
    printf("Hello message sent\n");
    sleep(5);
    return 0;
  }

  LOG(INFO) << "Eagerly loading files in memory...";

  // FIXME: remove, we should be loading files lazily
  //{
  //    auto load = [](string filename){
  //        // StorageManager::load(filename, PINNED);
  //        StorageManager::loadToCpus(filename);
  //    };
  //
  //
  // }

  LOG(INFO) << "Finished initialization";
  std::cout << "ready" << std::endl;
  std::string line;
  std::string prefix("--foo=");

  if (argc >= 2) {
    unlink_upon_exit uue;
    runPlanFile(argv[argc - 1], uue, true);
  } else {
    unlink_upon_exit uue;
    while (std::getline(std::cin, line)) {
      std::string cmd = trim(line);

      LOG(INFO) << "Command received: " << cmd;

      if (cmd == "quit") {
        std::cout << "quiting..." << std::endl;
        break;
      } else if (starts_with(cmd, "execute plan ")) {
        if (starts_with(cmd, "execute plan from file ")) {
          constexpr size_t prefix_size = clen("execute plan from file ");
          std::string plan = cmd.substr(prefix_size);
          std::string label = runPlanFile(plan, uue, echo);

          std::cout << "result in file /dev/shm/" << label << std::endl;
        } else {
          std::cout << "error (command not supported)" << std::endl;
        }
      } else if (starts_with(cmd, "echo")) {
        if (cmd == "echo results on") {
          echo = true;
        } else if (cmd == "echo results off") {
          echo = false;
        } else {
          std::cout << "error (unknown echo, please specify what to echo)"
                    << std::endl;
        }
      } else if (starts_with(cmd, "codegen")) {
        if (cmd == "codegen print on") {
          print_generated_code = true;
        } else if (cmd == "codegen print off") {
          print_generated_code = false;
        } else if (cmd == "codegen print query") {
          std::cout << print_generated_code << std::endl;
        } else {
          std::cout
              << "error (unknown codegen option, please specify what to echo)"
              << std::endl;
        }
      } else if (cmd == "unloadall") {
        StorageManager::getInstance().unloadAll();
        std::cout << "done" << std::endl;
      } else if (starts_with(cmd, "load ")) {
        // if (starts_with(cmd, "load locally ")){
        //     constexpr size_t prefix_size = clen("load locally ");
        //     std::string path             = cmd.substr(prefix_size);
        //     StorageManager::load(path, PINNED);
        //     std::cout << "done" << std::endl;
        // } else if (starts_with(cmd, "load cpus ")){
        //     constexpr size_t prefix_size = clen("load cpus ");
        //     std::string path             = cmd.substr(prefix_size);
        //     StorageManager::loadToCpus(path);
        //     std::cout << "done" << std::endl;
        // } else if (starts_with(cmd, "load gpus ")){
        //     constexpr size_t prefix_size = clen("load gpus ");
        //     std::string path             = cmd.substr(prefix_size);
        //     StorageManager::loadToGpus(path);
        //     std::cout << "done" << std::endl;
        // } else if (starts_with(cmd, "load localgpu ")){
        //     constexpr size_t prefix_size = clen("load localgpu ");
        //     std::string path             = cmd.substr(prefix_size);
        //     StorageManager::load(path, GPU_RESIDENT);
        //     std::cout << "done" << std::endl;
        // } else if (starts_with(cmd, "load everywhere ")){
        //     constexpr size_t prefix_size = clen("load everywhere ");
        //     std::string path             = cmd.substr(prefix_size);
        //     StorageManager::loadEverywhere(path);
        //     std::cout << "done" << std::endl;
        // } else {
        std::cout << "error (unknown load option, please specify where to load)"
                  << std::endl;
        // }
      }
    }
  }
  return 0;
}
