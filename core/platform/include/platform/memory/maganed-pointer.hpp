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

#ifndef PROTEUS_MAGANED_POINTER_HPP
#define PROTEUS_MAGANED_POINTER_HPP

#include <memory>
#include <string>

namespace proteus {
class internal_error : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};
}  // namespace proteus

template <typename T>
struct ManagedPointerDeleter {
  void operator()(T *ptr) const {
    if (!ptr) return;
    throw proteus::internal_error{"Leaked pointer " +
                                  std::to_string((uintptr_t)ptr)};
  }
};

template <typename T>
class ManagedUniquePtr {
 private:
  std::unique_ptr<T, ManagedPointerDeleter<T>> p;

 public:
  explicit ManagedUniquePtr(T *ptr) : p(ptr) {}
  constexpr ManagedUniquePtr(std::nullptr_t = nullptr) noexcept : p(nullptr) {}

  [[nodiscard]] inline auto get() const noexcept { return p.get(); }

  auto &operator=(std::nullptr_t) noexcept {
    p = nullptr;
    return *this;
  }

  auto operator<=>(const ManagedUniquePtr<T> &other) const {
    return p <=> other.p;
  }

  friend auto operator==(const ManagedUniquePtr<T> &po, std::nullptr_t) {
    return po.p == nullptr;
  }

  explicit operator bool() const noexcept { return (bool)p; }

  auto release() { return p.release(); }
};

namespace proteus {
template <typename T>
using managed = ManagedUniquePtr<T>;
using managed_ptr = managed<void>;
}  // namespace proteus

static_assert(
    sizeof(proteus::managed_ptr) == sizeof(void *),
    "Many modules assume that managed pointers can fit in normal pointers,"
    "do not break");
#endif  // PROTEUS_MAGANED_POINTER_HPP
