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

#ifndef PROTEUS_AFFINITIZATION_FACTORY_HPP
#define PROTEUS_AFFINITIZATION_FACTORY_HPP

#include <olap/operators/relbuilder.hpp>
#include <olap/routing/affinitizers.hpp>
#include <olap/routing/degree-of-parallelism.hpp>
#include <olap/routing/routing-policy-types.hpp>

class AffinitizationFactory {
 protected:
  bool isReduction(RelBuilder &input);

 public:
  virtual ~AffinitizationFactory();

  virtual DegreeOfParallelism getDOP(DeviceType trgt, RelBuilder &input) = 0;
  virtual RoutingPolicy getRoutingPolicy(DeviceType trgt, bool isHashBased,
                                         RelBuilder &input) = 0;
  virtual std::unique_ptr<Affinitizer> getAffinitizer(DeviceType trgt,
                                                      RoutingPolicy policy,
                                                      RelBuilder &input) = 0;

  virtual std::string getDynamicPgName(const std::string &relName) = 0;
};

#endif  // PROTEUS_AFFINITIZATION_FACTORY_HPP
