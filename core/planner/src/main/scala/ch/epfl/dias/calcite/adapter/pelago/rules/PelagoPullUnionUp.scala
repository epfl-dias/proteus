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

import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.traits.RelComputeDevice
import ch.epfl.dias.calcite.adapter.pelago.traits.RelComputeDeviceTraitDef
import ch.epfl.dias.calcite.adapter.pelago.traits.RelDeviceType
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHetDistribution
import org.apache.calcite.plan.RelOptRule
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.rel.RelNode
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRule.{any, convert, operand}

object PelagoPullUnionUp {
  val RULES: Array[RelOptRule] = Array[RelOptRule](
    new PelagoPullUnionUp(classOf[PelagoFilter]),
    new PelagoPullUnionUp(classOf[PelagoProject]),
    new PelagoPullUnionUp(classOf[PelagoAggregate]),
    new RelOptRule(
      operand(
        classOf[PelagoJoin],
        operand(classOf[PelagoUnion], any),
        operand(classOf[PelagoUnion], any)
      ),
      "PPUUPelagoJoin"
    ) {
      def fixDevice(e: RelNode): RelNode =
        convert(
          e,
          if (e.getTraitSet.contains(RelComputeDevice.X86_64))
            RelDeviceType.X86_64
          else RelDeviceType.NVPTX
        )

      def fix(e: RelNode, e1: RelNode): RelNode = {
        convert(
          convert(e, RelDeviceType.X86_64),
          e1.getTraitSet.getTrait(RelComputeDeviceTraitDef.INSTANCE)
        )
      }

      override def onMatch(call: RelOptRuleCall): Unit = {
        val rel: RelNode = call.rel(0)
        val ins0 = call.rel(1).asInstanceOf[RelNode].getInputs
        val ins1 = call.rel(2).asInstanceOf[RelNode].getInputs
        if (ins0.get(0).getTraitSet.contains(RelComputeDevice.X86_64NVPTX))
          return
        if (ins0.get(1).getTraitSet.contains(RelComputeDevice.X86_64NVPTX))
          return
        if (ins1.get(0).getTraitSet.contains(RelComputeDevice.X86_64NVPTX))
          return
        if (ins1.get(1).getTraitSet.contains(RelComputeDevice.X86_64NVPTX))
          return
        val v0 = fix(
          rel.copy(
            null,
            ImmutableList.of(
              convert(fixDevice(ins0.get(0)), RelHetDistribution.SPLIT_BRDCST),
              convert(fixDevice(ins1.get(0)), RelHetDistribution.SPLIT)
            )
          ),
          ins0.get(0)
        )
        call.transformTo(v0)
        val v1 = fix(
          rel.copy(
            null,
            ImmutableList.of(
              convert(fixDevice(ins0.get(1)), RelHetDistribution.SPLIT_BRDCST),
              convert(fixDevice(ins1.get(1)), RelHetDistribution.SPLIT)
            )
          ),
          ins0.get(1)
        )
        call.transformTo(v1)
        call.transformTo(
          PelagoUnion.create(ImmutableList.of(v0, v1), all = true)
        )
      }
    }
  )
}

class PelagoPullUnionUp protected (val op: Class[_ <: RelNode])
    extends RelOptRule(
      operand(op, operand(classOf[PelagoUnion], any)),
      "PPUU" + op.getName
    ) {
  override def onMatch(call: RelOptRuleCall): Unit = {
    val rel: RelNode = call.rel(0)
    val rel2: RelNode = call.rel(1)
    rel match {
      case aggregate: PelagoAggregate if aggregate.isGlobalAgg => return
      case _                                                   =>
    }
//    var ops = rel2.getInputs().stream().map((e) -> rel.copy(null,
//        ImmutableList.of(e)
//    ));
//
//    ops.allMatch(ops.)
    call.transformTo(
      PelagoUnion.create( //            rel2.getInputs().stream().map((e) -> {
//              var v = convert(convert(rel.copy(null,
////            ImmutableList.of(e)
//                  ImmutableList.of(convert(e,
//                      (e.getTraitSet().contains(RelComputeDevice.X86_64)) ?
//                          RelDeviceType.X86_64 :
//                          RelDeviceType.NVPTX
//                  ))
//              ), RelDeviceType.X86_64), e.getTraitSet().getTrait(RelComputeDeviceTraitDef.INSTANCE));
//              call.transformTo(v);
////          System.out.println(((VolcanoPlanner) call.getPlanner()).getSubset(v) + " \t\t\t\t" + rel);
//              return v;
//            })
//                .collect(Collectors.toList()),
        ImmutableList.of(
          convert(rel, rel2.getInput(0).getTraitSet),
          convert(rel, rel2.getInput(1).getTraitSet)
        ),
        all = true
      )
    )
  }
}
