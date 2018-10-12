package ch.epfl.dias.calcite.adapter.pelago.reporting;

import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;
import org.apache.calcite.sql2rel.NullInitializerExpressionFactory;

import java.sql.Timestamp;



public class TimeKeeperTable extends SqlCreateTable.MutableArrayTable {

    public static TimeKeeperTable INSTANCE = init();

    private TimeKeeperTable(String name, RelProtoDataType protoStoredRowType,
                    RelProtoDataType protoRowType,
                    InitializerExpressionFactory initializerExpressionFactory) {
        super(name, protoStoredRowType, protoRowType, initializerExpressionFactory);
    }

    private static TimeKeeperTable init() {
        JavaTypeFactory typeFactory = new JavaTypeFactoryImpl();

        RelDataTypeFactory.Builder sb = typeFactory.builder();
        sb.add("total_time", SqlTypeName.BIGINT);
        sb.add("planning_time", SqlTypeName.BIGINT);
        sb.add("plan2json_time", SqlTypeName.BIGINT);
        sb.add("executor_time", SqlTypeName.BIGINT);
        sb.add("codegen_time", SqlTypeName.BIGINT);
        sb.add("execution_time", SqlTypeName.BIGINT);
        sb.add("timestamp", SqlTypeName.VARCHAR);

        InitializerExpressionFactory ief = new NullInitializerExpressionFactory();

        return new TimeKeeperTable("Timings", RelDataTypeImpl.proto(sb.build()), RelDataTypeImpl.proto(sb.build()), ief);
    }

    public static void addTimings(long ttotal_ms, long tplanning_ms, long tplan2json_ms, long texecutor_ms,
            long tcodegen_ms, long texecution_ms, Timestamp timestamp){
        Object[] arr = {ttotal_ms, tplanning_ms, tplan2json_ms, texecutor_ms, tcodegen_ms, texecution_ms, timestamp.toString()};
        INSTANCE.rows.add(arr);
    }
}
