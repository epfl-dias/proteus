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

#include <lib/examples.hpp>

/**
 * This file shows the basic RelBuilder API using single-threaded plan,
 * a multi-threaded plan and a function to show that partial plans are
 * composable.
 *
 * The RelBuilder uses the Fluent API and serializing/deserializing a
 * JSON plan happens with a switch statement that checks the
 * "operator" field and calls the corresponding function.
 */

/**
 * Generate a simple plan to demonstrate the usage of the RelBuilder API
 *
 * Plans are generally produced through Proteus' query optimizer and both
 * operators and expressions are serialized as JSONs.
 *
 * (Proteus' query optimizer is build on top of Calcite)
 */
RelBuilder generateSingleThreadedPlan(RelBuilder builder,
                                      CatalogParser &catalog) {
  return builder
      /**
       * All (global) plans start with scans (or values).
       * Scans read the metadata from the catalogs and allow data format
       * plugins to inject their logic for interpreting data.
       * The latter happens by the plugin registering itself as the source
       * of tuples. Then during expression evaluation the source of each
       * input is invoked to determine how a field is processed.
       *
       * Lastly, the scan segments input relations.
       * If the plugin outputs blocks, the segmentation is just pointer
       * arithmetics and it does not touch the data (example plugins: block).
       * If the plugin outputs tuples, the segmentation is producing these
       * tuples and thus requires accessing data (example plugins: pm-csv).
       *
       * @param   relName   The name of the table we access. In general, this is
       *                    an arbitrary string, but for files fetched using the
       *                    same path across all servers, it can be that path.
       *                    The file name extension provides no information
       *                    the plugin that will be used for this table.
       *
       *                    For example, the name may end with .csv for a file
       *                    generated from a csv file as input, but containing
       *                    the binary data in columnar formats.
       *                    The "block" plugin is Proteus default plugin for
       *                    high-performance analytics and it signifies that the
       *                    data are in binary columnar format.
       *                    The plugin will use the relName as teh basis for
       *                    finding the rest of the column, but the actual files
       *                    will be "inputs/ssbm100/date.bin.d_datekey" etc.
       *
       * @param   relAttrs  The columns participating in this query.
       *                    For hierarchical data, this would usually be the
       *                    top level columns and unnest will follow for inner
       *                    attributes.
       *                    In general any plugin is allowed to interpret the
       *                    attribute names as it prefers.
       *
       * @param   catalog   Both the scan and the plugin may require to fetch
       *                    extra information from the catalog, as for example
       *                    the rest of the participating columns, statistics
       *                    and/or plugin-specific information (like which
       *                    global partitions is on this server for this rel)
       *
       * @param   pg        The plugin type that will be used for this relation.
       *                    Example plugins are "block", "json", "pm-csv",
       *                    "distributed-block".
       *                    The plugin name is used to locate the factory
       *                    function for instantiating the plugin.
       *                    If the name is a "::"-separated list of strings,
       *                    the last string is interpreted as the plugin type,
       *                    the string before the last is interpreted as the
       *                    dynamic library name and any additional strings
       *                    will cause the 1-to-(N-1) strings to be interpreted
       *                    as a path.
       *
       * @see Input plugins: Karpathiotakis et al, VLDB2016
       */
      .scan(
          /* relName */ "inputs/ssbm100/date.bin",
          /* relAttrs */ {"d_datekey", "d_year"},
          /* catalog */ catalog,
          /* pg */ pg{"block"})
      /**
       * Pull data into the current NUMA node, if not already there.
       *
       * @param     slack   slack between the two ends of the memmove pipe,
       *                    used for load balancing
       *                    (limits the on-the-fly transfers)
       * @param     to      the type of the current NUMA node
       *
       * @see Input plugins: Chrysogelos et al, VLDB2019
       */
      .memmove(8, DeviceType::CPU)
      /**
       * Unpack iterates over the tuples of each input block.
       *
       * For example, if a scan above the unpack generates block of d_year and
       * d_datekey, then the unpack will iterate over these blocks and it will
       * unpack each block into tuples{d_year, d_datekey}.
       *
       * @see Input plugins: Chrysogelos et al, VLDB2019
       */
      .unpack()
      /**
       * Filters tuples using a predicate.
       *
       * Yields each tuple for which the predicate is true, to the next
       * operator.
       *
       * @param     pred    Predicate used for filtering.
       */
      .filter(
          /* pred */
          [&](const auto &arg) -> expression_t {
            /* (d_datekey mod 30) > 15 */
            return gt(arg["d_datekey"] % 30, 15);
          })
      /**
       * Performs simple aggregations with a single group.
       *
       * @param     expr    Inputs to the (monoid) aggregates
       * @param     accs    Aggregate type (ie. SUM for a summation).
       *                    The attribute name of an aggregate is the same as
       *                    the input name.
       */
      .reduce(
          /* expr */
          [&](const auto &arg) -> std::vector<expression_t> {
            return {/* (d_year * d_year) as year_sq_sum */
                    (arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
                    /* (d_year) as year_sum */
                    arg["d_year"].as("tmp", "year_sum"),
                    /* (1) as cnt */
                    expression_t{1}.as("tmp", "cnt")};
          },
          /* accs */
          {SUM, SUM, SUM})
      /**
       * Project (calculate) tuple-wise expressions.
       *
       * @param     expr    List of expressions to evaluate.
       *
       * @note  expressions usually perserve types and they look like c++
       *        default expression in terms of type convergences, without
       *        c++'s defeult automatic casts.
       */
      .project(
          /* expr */
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["cnt"].as("out", "cnt"),
                    (arg["year_sum"] / arg["cnt"]).as("out", "avg"),
                    ((arg["year_sq_sum"] / arg["cnt"]) -
                     ((arg["year_sum"] / arg["cnt"]) *
                      (arg["year_sum"] / arg["cnt"])))
                        .as("out", "var")};
          })
      /**
       * Print the results using the given plugin.
       *
       * @param     pgType  Plugin to be used for output.
       */
      .print(/* pgType */ pg{"pm-csv"});
}

/**
 * Generate a simple parallel plan to demonstrate the usage of the RelBuilder
 *
 * Proteus' query optimizer auto-parallelizes the plans using the appropriate
 * rules.
 */
RelBuilder generateMultiThreadedPlan(RelBuilder builder,
                                     CatalogParser &catalog) {
  return builder
      .scan("inputs/ssbm100/date.bin", {"d_datekey", "d_year"}, catalog,
            pg{"block"})
      /**
       * Router distributes the inputs to @p fanout workers and handles the
       * affinity of the workers.
       *
       * How inputs are distributed across workers depends on the
       * RoutingPolicies.
       * Affinity of the workers depends on the target and the affinitization
       * policy. The default affinitization policy does a round-robin across
       * NUMA nodes.
       *
       * @see Input plugins: Chrysogelos et al, VLDB2019
       */
      .router(
          /* fanout */ DegreeOfParallelism{topology::getInstance()
                                               .getCoreCount()},
          /* slack */ 8,
          /* p */ RoutingPolicy::LOCAL, /* target */ DeviceType::CPU)
      /*
       * As the router above has a LOCAL policy, it not cause socket-to-socket
       * transfers.
       */
      .memmove(8, DeviceType::CPU)
      /*
       * The router above distributed blocks, now we need to unpack them.
       *
       * Operators following the router are instantiated multiple times,
       * once per worker.
       * That is, for the above router that had a fanout == #cores,
       * we will have #cores unpack instances.
       */
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg["d_datekey"] % 30, 15);
      })
      /*
       * The router above distributed blocks, so this reduce will provide
       * worker-local results, we need another router afterwards to
       * combine them.
       */
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
                    arg["d_year"].as("tmp", "year_sum"),
                    expression_t{1}.as("tmp", "cnt")};
          },
          {SUM, SUM, SUM})
      /* gather the partial aggregates into one worker */
      .router(
          /* fanout */ DegreeOfParallelism{1},
          /* slack */ 8,
          /* p */ RoutingPolicy::RANDOM, /* target */ DeviceType::CPU)
      /*
       * Now we run with one worker, based on the above DOP, so this reduce
       * will produce the global result
       */
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["year_sq_sum"], arg["year_sum"], arg["cnt"]};
          },
          {SUM, SUM, SUM})
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {
            arg["cnt"].as("out", "cnt"),
            (arg["year_sum"] / arg["cnt"]).as("out", "avg"),
            ((arg["year_sq_sum"] / arg["cnt"]) -
             ((arg["year_sum"] / arg["cnt"]) * (arg["year_sum"] / arg["cnt"])))
                .as("out", "var")};
      })
      /**
       * Print the results using the given plugin.
       *
       * @param     pgType  Plugin to be used for output.
       */
      .print(/* pgType */ pg{"pm-csv"});
}

RelBuilder generateLocalReductionTask(RelBuilder builder) {
  return builder.unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg["d_datekey"] % 30, 15);
      })
      /*
       * The router above distributed blocks, so this reduce will provide
       * worker-local results, we need another router afterwards to
       * combine them.
       */
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
                    arg["d_year"].as("tmp", "year_sum"),
                    expression_t{1}.as("tmp", "cnt")};
          },
          {SUM, SUM, SUM});
}

RelBuilder naivelyParallelize(RelBuilder builder,
                              const std::function<RelBuilder(RelBuilder)> &f) {
  auto preparallel =
      builder
          .router(DegreeOfParallelism{topology::getInstance().getCoreCount()},
                  8, RoutingPolicy::LOCAL, DeviceType::CPU)
          /*
           * As the router above has a LOCAL policy, it not cause
           * socket-to-socket transfers.
           */
          .memmove(8, DeviceType::CPU);
  return f(preparallel)
      .router(DegreeOfParallelism{1}, 8, RoutingPolicy::RANDOM,
              DeviceType::CPU);
}

/**
 * Generate a simple parallel plan to demonstrate the usage of the RelBuilder
 * using multiple subtasks.
 *
 * Proteus' query optimizer auto-parallelizes the plans using the appropriate
 * rules.
 */
RelBuilder generatePlanComposition(RelBuilder builder, CatalogParser &catalog) {
  /* Declare what we want to read */
  auto input = builder.scan("inputs/ssbm100/date.bin", {"d_datekey", "d_year"},
                            catalog, pg{"block"});

  /* Declare subtask (local reduction) over the input */
  auto subtaskDescription = generateLocalReductionTask;

  /* Parallelize subtask */
  auto partialReductions = naivelyParallelize(input, subtaskDescription);

  /* Compute final aggregation */
  auto globalReduction =
      partialReductions
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["year_sq_sum"], arg["year_sum"], arg["cnt"]};
              },
              {SUM, SUM, SUM})
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["cnt"].as("out", "cnt"),
                    (arg["year_sum"] / arg["cnt"]).as("out", "avg"),
                    ((arg["year_sq_sum"] / arg["cnt"]) -
                     ((arg["year_sum"] / arg["cnt"]) *
                      (arg["year_sum"] / arg["cnt"])))
                        .as("out", "var")};
          })
          /**
           * Print the results using the given plugin.
           *
           * @param     pgType  Plugin to be used for output.
           */
          .print(/* pgType */ pg{"pm-csv"});

  return globalReduction;
}
