/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_ERASE_CONSTRUCTOR_IDIOMS_HPP
#define PROTEUS_ERASE_CONSTRUCTOR_IDIOMS_HPP

namespace proteus::utils {

/*
 Move constructors/assignment
   T(T &&)
   T& T::operator=(T &&)

 Copy constructors/assignment
   T(const T &)
   T &operator=(const T &)
*/

class remove_copy;
class remove_move;
class remove_copy_move;

class remove_copy {
 public:
  remove_copy(const remove_copy &) = delete;
  remove_copy &operator=(const remove_copy &) = delete;

  remove_copy() = default;
  remove_copy(remove_copy &&) = default;
  remove_copy &operator=(remove_copy &&) = default;
};

class remove_move {
 public:
  remove_move(remove_move &&) = delete;
  remove_move &operator=(remove_move &&) = delete;

  remove_move() = default;
  remove_move(const remove_move &) = default;
  remove_move &operator=(const remove_move &) = default;
};

class remove_copy_move {
 public:
  remove_copy_move(const remove_copy_move &) = delete;
  remove_copy_move &operator=(const remove_copy_move &) = delete;
  remove_copy_move(remove_copy_move &&) = delete;
  remove_copy_move &operator=(remove_copy_move &&) = delete;

  remove_copy_move() = default;
};

}  // namespace proteus::utils

#endif  // PROTEUS_ERASE_CONSTRUCTOR_IDIOMS_HPP
