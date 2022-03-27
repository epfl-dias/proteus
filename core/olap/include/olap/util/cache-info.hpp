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

#ifndef PROTEUS_CACHE_INFO_HPP
#define PROTEUS_CACHE_INFO_HPP

#include <olap/values/expressionTypes.hpp>

struct CacheInfo {
  /* XXX Issue: Types depend on context
   * Do I need to share the context for this to work?
   * One option is mapping LLVM Types to an enum -
   * But what about custom types?
   * An option is that everything considered
   * a custom / structType will be stored in CATALOG too */
  // StructType *objectType;
  std::list<typeID> objectTypes;
  /* Convention:
   * Count begins from 1.
   * Zero implies we're dealing with the whole obj. (activeLoop)
   * Negative implies invalid entry */
  int structFieldNo;
  // Pointers to facilitate LLVM storing stuff in them
  char **payloadPtr;
  size_t *itemCount;
};

#endif /* PROTEUS_CACHE_INFO_HPP */
