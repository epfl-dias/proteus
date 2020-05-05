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

#include <gflags/gflags.h>

#include <common/olap-common.hpp>
#include <cstring>

#include "cli-flags.hpp"
#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "plan/prepared-statement.hpp"
#include "storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

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
  return PreparedStatement::from(planPath, label, catalogJSON).execute();
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
        StorageManager::unloadAll();
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
