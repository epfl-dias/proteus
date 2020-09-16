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

#ifndef PROTEUS_TYPES_HPP
#define PROTEUS_TYPES_HPP

#include <cstdint>
#include <variant>
#include <vector>

// FIXME: should/can we generate that based on the ExpressionTypes?
//  Does it make sense? This is specific to each plugin
typedef std::variant<int32_t, int64_t, double> proteus_any;

template <typename A>
struct variant_to_vector_variant_helper;

template <typename... A>
struct variant_to_vector_variant_helper<std::variant<A...>> {
  using type = std::variant<std::vector<A>...>;
};

typedef variant_to_vector_variant_helper<proteus_any>::type proteus_any_vector;

#endif  // PROTEUS_TYPES_HPP
