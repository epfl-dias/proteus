/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_RELBUILDER_FACTORY_HPP
#define PROTEUS_RELBUILDER_FACTORY_HPP

#include <operators/relbuilder.hpp>

class RelBuilderFactory {
 private:
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  explicit RelBuilderFactory(std::string name);
  ~RelBuilderFactory();  // Required for std::unique_ptr, as impl is incomplete

  [[nodiscard]] RelBuilder getBuilder() const;
};

#endif /* PROTEUS_RELBUILDER_FACTORY_HPP */
