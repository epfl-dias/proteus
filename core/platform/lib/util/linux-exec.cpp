/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <array>
#include <cstdio>
#include <platform/util/linux-exec.hpp>
#include <string>

std::pair<std::string, int> execCommand(const std::string &cmd) {
  int exit_status = 0;
  auto pPipe = ::popen(cmd.c_str(), "r");
  if (pPipe == nullptr) {
    throw std::runtime_error("Cannot open pipe");
  }

  std::array<char, 256> buffer{};

  std::string result;

  while (not std::feof(pPipe)) {
    auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
    result.append(buffer.data(), bytes);
  }

  auto rc = ::pclose(pPipe);

  if (WIFEXITED(rc)) {
    exit_status = WEXITSTATUS(rc);
  }

  return std::make_pair(result, exit_status);
}
