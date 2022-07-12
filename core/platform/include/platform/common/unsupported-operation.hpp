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

#ifndef PROTEUS_UNSUPPORTED_OPERATION_HPP
#define PROTEUS_UNSUPPORTED_OPERATION_HPP

#include <exception>
#include <sstream>
#include <string>

namespace proteus {
class unsupported_operation : public std::exception {
  // shared_ptr as exceptions are not allowed to throw during copy
  std::shared_ptr<std::string> msg;

 public:
  unsupported_operation(std::string msg)
      : msg(std::make_shared<std::string>(std::move(msg))) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg->c_str();
  }
};

class abort : public std::runtime_error {
 public:
  abort() : runtime_error("tx abort") {}
};

class runtime_error : public std::runtime_error {
 protected:
  explicit runtime_error(std::string msg)
      : std::runtime_error(std::move(msg)) {}

 public:
  runtime_error() : runtime_error("") {}

  template <typename T>
  [[nodiscard]] inline runtime_error operator<<(const T& v) {
    std::stringstream ss;
    ss << what() << v;
    return runtime_error{ss.str()};
  }
};
}  // namespace proteus

#endif  // PROTEUS_UNSUPPORTED_OPERATION_HPP
