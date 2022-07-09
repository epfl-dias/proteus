/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#ifndef PROTEUS_IMEMFILE_HPP
#define PROTEUS_IMEMFILE_HPP

#include <storage/storage-manager.hpp>

namespace proteus {

template <typename T>
class imemfile {
  class voidimemfile {
   protected:
    FileRequest req;
    mem_file mf;

   private:
    static auto getSingleMemFile(FileRequest &req) {
      req.pin();
      assert(req.getSegmentCount() == 1);
      return req.getSegments().at(0);
    }

   public:
    inline explicit voidimemfile(const std::string &path)
        : req(StorageManager::getInstance().request(path, 1, PAGEABLE)),
          mf(getSingleMemFile(req)) {}

    [[nodiscard]] size_t size() const { return mf.size; }
    [[nodiscard]] auto data() const { return mf.data; }

    inline ~voidimemfile() { req.unpin(); }
  };

  voidimemfile file;

 public:
  explicit imemfile(const std::string &path) : file(path) {}

  [[nodiscard]] inline size_t size() const { return file.size() / sizeof(T); }

  [[nodiscard]] inline auto data() const {
    return static_cast<const T *>(file.data());
  }

  [[nodiscard]] inline auto &operator[](size_t i) const {
    assert(i < size());
    return data()[i];
  }

  [[nodiscard]] inline auto &operator[](size_t i) {
    assert(i < size());
    return data()[i];
  }
};
}  // namespace proteus

#endif  // PROTEUS_IMEMFILE_HPP
