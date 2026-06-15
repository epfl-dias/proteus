/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_TOPOLOGY_PARSER_HPP
#define PROTEUS_TOPOLOGY_PARSER_HPP

#include <fstream>
#include <sstream>
#include <vector>

class ThreadSiblingParser {
 public:
  // A thread list file contains a human
  // readable list of thread IDs.
  // Example: 0-8,18-26
  // https://www.kernel.org/doc/Documentation/cputopology.txt

  static std::vector<uint32_t> getThreadSiblings(uint32_t core_id) {
    auto threadList = getString(getThreadFileName(core_id));
    auto tokens = split(threadList, ',');

    std::vector<uint32_t> ret;

    for (auto& str : tokens) {
      auto values = split(str, '-');
      if (values.size() == 1) {
        ret.push_back(stoul(values[0]));
        // threads++;
      } else {
        auto t0 = stoul(values.at(0));
        auto t1 = stoul(values.at(1));
        for (size_t i = t0; i <= t1; i++) {
          ret.push_back(i);
        }
      }
    }
    return ret;
  }

 private:
  static inline std::string getThreadFileName(uint32_t core_id) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(core_id) +
           "/topology/thread_siblings_list";
  }

  static inline std::vector<std::string> split(const std::string& str,
                                               char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);

    while (getline(tokenStream, token, delimiter)) tokens.push_back(token);

    return tokens;
  }

  static inline std::string getString(const std::string& filename) {
    std::ifstream file(filename);
    std::string str;

    // Read the first std::string,
    // stops at any space character
    if (file && (file >> str))
      return str;
    else
      return {};
  }
};

class CorePackageParser {
 public:
  static uint32_t getCorePackageId(uint32_t core_id) {
    std::ifstream s(getCorePackagePath(core_id));
    PCHECK(s.is_open()) << "Failed to open: " << getCorePackagePath(core_id);
    uint32_t package_id;
    s >> package_id;
    return package_id;
  }

  static inline std::string getCorePackagePath(uint32_t core_id) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(core_id) +
           "/topology/physical_package_id";
  }
};

#endif  // PROTEUS_TOPOLOGY_PARSER_HPP
