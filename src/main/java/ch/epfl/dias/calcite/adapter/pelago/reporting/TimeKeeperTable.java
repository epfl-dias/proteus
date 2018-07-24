package ch.epfl.dias.calcite.adapter.pelago.reporting;

import com.google.common.base.Preconditions;
import org.apache.calcite.DataContext;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.linq4j.*;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.logical.LogicalTableModify;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeImpl;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.*;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.schema.impl.AbstractTableQueryable;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;
import org.apache.calcite.sql2rel.NullInitializerExpressionFactory;

import java.lang.reflect.Type;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collection;
import java.util.List;



public class TimeKeeperTable extends SqlCreateTable.MutableArrayTable {

    public static TimeKeeperTable INSTANCE;

    private TimeKeeperTable(String name, RelProtoDataType protoStoredRowType,
                    RelProtoDataType protoRowType,
                    InitializerExpressionFactory initializerExpressionFactory) {
        super(name, protoStoredRowType, protoRowType, initializerExpressionFactory);
    }

    private static TimeKeeperTable init() {
        JavaTypeFactory typeFactory = new JavaTypeFactoryImpl();

        RelDataTypeFactory.Builder sb = typeFactory.builder();
        sb.add("time_calcite", SqlTypeName.BIGINT);
        sb.add("time_exec", SqlTypeName.BIGINT);
        sb.add("time_codegen", SqlTypeName.BIGINT);
        sb.add("timestamps", SqlTypeName.VARCHAR);

        InitializerExpressionFactory ief = new NullInitializerExpressionFactory();

        return new TimeKeeperTable("PelagoTimeKeeper", RelDataTypeImpl.proto(sb.build()), RelDataTypeImpl.proto(sb.build()), ief);
    }

    public static TimeKeeperTable getInstance(){
        if(INSTANCE == null) {
            INSTANCE = init();
        }

        return INSTANCE;
    }

    public static void addTimings(long texec, long tcodegen, long tcalcite, Timestamp timestamp){
        Object[] arr = {tcalcite, texec, tcodegen, timestamp.toString()};
        INSTANCE.rows.add(arr);
    }
}
