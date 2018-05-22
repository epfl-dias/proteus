package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptAbstractTable;
import org.apache.calcite.plan.RelOptSchema;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.type.RelDataType;

public class PelagoRelOptTable extends RelOptAbstractTable {

    public PelagoRelOptTable(RelOptSchema schema, String name, RelDataType rowType){
        super(schema, name, rowType);
    }

    public RelDistribution getDistribution(){
        return RelDistributions.RANDOM_DISTRIBUTED;
    }
}
