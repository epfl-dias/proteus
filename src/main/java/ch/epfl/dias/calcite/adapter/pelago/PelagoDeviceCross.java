//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceType;
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceTypeTraitDef;
//import ch.epfl.dias.emitter.Binding;
//import com.google.common.base.Supplier;
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.RelWriter;
//import org.apache.calcite.rel.SingleRel;
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//public class PelagoDeviceCross extends SingleRel implements PelagoRel{
//    private final RelDeviceType toDevice;
//
//    protected PelagoDeviceCross(
//            RelOptCluster cluster,
//            RelTraitSet traits,
//            RelNode input,
//            RelDeviceType toDevice) {
//        super(cluster, traits, input);
//        this.toDevice = toDevice;
//    }
//
//    public static PelagoDeviceCross create(RelNode input, RelDeviceType toDevice){
//        RelOptCluster cluster  = input.getCluster();
//        RelTraitSet   traitSet = input.getTraitSet().replace(PelagoRel.CONVENTION)
//                                    .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
//            public RelDeviceType get() { return toDevice; }
//        });
//        return new PelagoDeviceCross(input.getCluster(), traitSet, input, toDevice);
//    }
//
//    @Override
//    public RelWriter explainTerms(RelWriter pw) {
//        return super.explainTerms(pw).item("to", toDevice);
//    }
//
//    @Override
//    public Tuple2<Binding, JsonAST.JValue> implement() {
//        return null;
//    }
//}
