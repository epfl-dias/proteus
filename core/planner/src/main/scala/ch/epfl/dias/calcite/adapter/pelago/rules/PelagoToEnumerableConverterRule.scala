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

package ch.epfl.dias.calcite.adapter.pelago.rules

import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories
import ch.epfl.dias.calcite.adapter.pelago.rel.{
  PelagoRel,
  PelagoToEnumerableConverter
}
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelComputeDevice,
  RelDeviceType,
  RelDeviceTypeTraitDef,
  RelHetDistribution,
  RelHomDistribution
}
import com.google.common.collect.ImmutableList
import org.apache.calcite.adapter.enumerable.EnumerableConvention
import org.apache.calcite.plan.RelRule.Config
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.tools.RelBuilderFactory
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.hint.{Hintable, RelHint}

/**
  * Rule to convert a relational expression from
  * [[PelagoRel.CONVENTION]] to [[EnumerableConvention]].
  */
object PelagoToEnumerableConverterRule {
  val INSTANCE = new PelagoToEnumerableConverterRule(
    PelagoRelFactories.PELAGO_BUILDER
  )
}

/**
  * Creates a CassandraToEnumerableConverterRule.
  *
  * @param relBuilderFactory Builder for relational expressions
  */
class PelagoToEnumerableConverterRule(relBuilderFactory: RelBuilderFactory)
    extends ConverterRule(
      Config.EMPTY
        .withRelBuilderFactory(relBuilderFactory)
        .as(classOf[ConverterRule.Config])
        .withConversion(
          classOf[RelNode],
          PelagoRel.CONVENTION,
          EnumerableConvention.INSTANCE,
          "PelagoToEnumerableConverterRule"
        )
    ) {

  override def convert(rel: RelNode): RelNode = { //        RelTraitSet newTraitSet = rel.getTraitSet().replace(getOutConvention()); //.replace(RelDeviceType.ANY);
//        RelNode inp = rel;
//        RelNode inp = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), rel, RelDistributions.SINGLETON, true);
//        RelNode inp = LogicalExchange.create(rel, RelDistributions.SINGLETON);
//        System.out.println(inp.getTraitSet());
    val traitSet = rel.getTraitSet
      .replace(PelagoRel.CONVENTION)
      .replace(RelHomDistribution.SINGLE)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replace(RelComputeDevice.X86_64NVPTX)
      .replace(RelHetDistribution.SINGLETON)
    PelagoToEnumerableConverter.create(
      RelOptRule.convert(rel, traitSet),
      rel match {
        case hintable: Hintable => hintable.getHints
        case _                  => ImmutableList.of[RelHint]
      }
    )
  }

  override def matches(call: RelOptRuleCall): Boolean = { //        return true;
//        if (!call.rel(0).getTraitSet().satisfies(RelTraitSet.createEmpty().plus(RelDistributions.SINGLETON))) return false;
//        if (!call.rel(0).getTraitSet().contains(RelDeviceType.X86_64)) return false;
//        if (call.rel(0).getTraitSet().containsIfApplicable(PelagoRel.CONVENTION())) return false;
    call
      .rel(0)
      .asInstanceOf[RelNode]
      .getTraitSet
      .containsIfApplicable(RelHetDistribution.SINGLETON)
  }
}
