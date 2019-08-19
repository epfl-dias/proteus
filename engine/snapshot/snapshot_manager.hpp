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

#include <cstdlib>

class arena_t {
  arena_t(const arena_t&) = delete;
  arena_t(arena_t&&) = delete;
  arena_t& operator=(const arena_t&) = delete;
  arena_t& operator=(arena_t&&) = delete;

  virtual void create_snapshot();
  virtual void destroy_snapshot();

  virtual void* oltp() const;
  virtual void* olap() const;
};

template <typename T>
class SnapshotManager_impl {
 public:
  static void init();
  static void destroy();

  static T create(size_t bytes) { return {}; }
  static void destroy(arena_t&& arena) {}

 private:
};

typedef SnapshotManager_impl<arena_t> SnapshotManager;
