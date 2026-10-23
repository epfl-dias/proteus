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

#ifndef PROTEUS_ROUTING_POLICY_TYPES_HPP
#define PROTEUS_ROUTING_POLICY_TYPES_HPP

enum class RoutingPolicy { RANDOM, LOCAL, FORCE_LOCAL, HASH_BASED };

/**
 * @see GeneralizedRouter
 */
enum class GeneralizedRoutingPolicy {
  SHARED_RANDOM,  /// Random across all consumers and without regard for data
                  /// locality
  SHARED_LOCAL,   /// Random across all consumers but with regard for data
                  /// locality
  SHARED_FORCE_LOCAL,  /// For now the same as SHARED_LOCAL
  SHARED_HASH_BASED,  /// Hash-based routing with consumers sharing queues. This
                      /// is not implemented
  DISTINCT_RANDOM_SPLIT_DATA_LOCAL  /// Random across consumers but with regard
                                    /// for data locality. Unlike SHARED_LOCAL,
                                    /// this policy uses separate queues per
                                    /// consumer
};

#endif  // PROTEUS_ROUTING_POLICY_TYPES_HPP
