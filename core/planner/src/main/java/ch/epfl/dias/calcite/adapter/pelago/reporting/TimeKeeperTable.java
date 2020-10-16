package ch.epfl.dias.calcite.adapter.pelago.reporting;

import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.server.PelagoMutableArrayTable;
import org.apache.calcite.sql.ddl.SqlCreatePelagoTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;
import org.apache.calcite.sql2rel.NullInitializerExpressionFactory;

import java.sql.Timestamp;



public class TimeKeeperTable extends PelagoMutableArrayTable {

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
        sb.add("dataload_time", SqlTypeName.BIGINT);
        sb.add("code_opt_time", SqlTypeName.BIGINT);
        sb.add("code_optnload_time", SqlTypeName.BIGINT);
        sb.add("execution_time", SqlTypeName.BIGINT);
        sb.add("timestamp", SqlTypeName.VARCHAR);
        sb.add("query", SqlTypeName.VARCHAR);
        sb.add("cmd_type", SqlTypeName.VARCHAR);
        sb.add("hwmode", SqlTypeName.VARCHAR);
        sb.add("plan", SqlTypeName.VARCHAR);

        InitializerExpressionFactory ief = new NullInitializerExpressionFactory();

        return new TimeKeeperTable("Timings", RelDataTypeImpl.proto(sb.build()), RelDataTypeImpl.proto(sb.build()), ief);
    }

    public static void addTimings(long ttotal_ms, long tplanning_ms,
            long tplan2json_ms, long texecutor_ms, long tcodegen_ms,
            long tdataload_ms,
            long tcode_opt_time_ms, long tcode_optnload_time_ms,
            long texecution_ms, Timestamp timestamp,
            String query,
            String cmd_type,
            String hwmode,
            String plan){
        Object[] arr = {ttotal_ms, tplanning_ms, tplan2json_ms, texecutor_ms, tcodegen_ms, tdataload_ms,
            tcode_opt_time_ms, tcode_optnload_time_ms, texecution_ms, timestamp.toString(),
            query, cmd_type, hwmode, plan};
        INSTANCE.getModifiableCollection().add(arr);
    }
}
