package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.rel.*;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelComputeDeviceTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHetDistribution;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

public class PelagoPullUnionUp extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    new PelagoPullUnionUp(PelagoFilter.class),
    new PelagoPullUnionUp(PelagoProject.class),
    new PelagoPullUnionUp(PelagoAggregate.class),
    new RelOptRule(operand(PelagoJoin.class, operand(PelagoUnion.class, any()), operand(PelagoUnion.class, any())), "PPUUPelagoJoin") {

      public RelNode fixDevice(RelNode e) {
        return convert(e,
            (e.getTraitSet().contains(RelComputeDevice.X86_64)) ?
                RelDeviceType.X86_64 :
                RelDeviceType.NVPTX
        );
      }

      public RelNode fix(RelNode e, RelNode e1) {
        return convert(convert(e, RelDeviceType.X86_64), e1.getTraitSet().getTrait(RelComputeDeviceTraitDef.INSTANCE));
      }

      @Override
      public void onMatch(final RelOptRuleCall call) {
        var rel = call.rel(0);
        var ins0 = call.rel(1).getInputs();
        var ins1 = call.rel(2).getInputs();
        if (ins0.get(0).getTraitSet().contains(RelComputeDevice.X86_64NVPTX)) return;
        if (ins0.get(1).getTraitSet().contains(RelComputeDevice.X86_64NVPTX)) return;
        if (ins1.get(0).getTraitSet().contains(RelComputeDevice.X86_64NVPTX)) return;
        if (ins1.get(1).getTraitSet().contains(RelComputeDevice.X86_64NVPTX)) return;

        var v0 = fix(
            rel.copy(null,
                ImmutableList.of(
                    convert(
                        fixDevice(ins0.get(0)),
                        RelHetDistribution.SPLIT_BRDCST
                    ),
                    convert(
                        fixDevice(ins1.get(0)),
                        RelHetDistribution.SPLIT
                    )
                )
            ), ins0.get(0));

        call.transformTo(v0
        );

        var v1 = fix(
            rel.copy(null,
                ImmutableList.of(
                    convert(
                        fixDevice(ins0.get(1)),
                        RelHetDistribution.SPLIT_BRDCST
                    ),
                    convert(
                        fixDevice(ins1.get(1)),
                        RelHetDistribution.SPLIT
                    )
                )
            ), ins0.get(1));

        call.transformTo(v1
        );

        call.transformTo(PelagoUnion.create(ImmutableList.of(v0, v1), true)
        );
      }
    }
  };

  protected PelagoPullUnionUp(Class<? extends RelNode> op) {
    super(operand(op, operand(PelagoUnion.class, any())), "PPUU" + op.getName());
  }

  public void onMatch(RelOptRuleCall call) {
    var rel = call.rel(0);
    var rel2 = call.rel(1);

    if (rel instanceof PelagoAggregate && ((PelagoAggregate) rel).isGlobalAgg()) return;

//    var ops = rel2.getInputs().stream().map((e) -> rel.copy(null,
//        ImmutableList.of(e)
//    ));
//
//    ops.allMatch(ops.)

    call.transformTo(
        PelagoUnion.create(
//            rel2.getInputs().stream().map((e) -> {
//              var v = convert(convert(rel.copy(null,
////            ImmutableList.of(e)
//                  ImmutableList.of(convert(e,
//                      (e.getTraitSet().contains(RelComputeDevice.X86_64)) ?
//                          RelDeviceType.X86_64 :
//                          RelDeviceType.NVPTX
//                  ))
//              ), RelDeviceType.X86_64), e.getTraitSet().getTrait(RelComputeDeviceTraitDef.INSTANCE));
//
//
//              call.transformTo(v);
//
////          System.out.println(((VolcanoPlanner) call.getPlanner()).getSubset(v) + " \t\t\t\t" + rel);
//
//              return v;
//            })
//                .collect(Collectors.toList()),

            ImmutableList.of(
              convert(rel, rel2.getInput(0).getTraitSet()),
              convert(rel, rel2.getInput(1).getTraitSet())
            ),
            true
        )
    )
    ;
  }
}