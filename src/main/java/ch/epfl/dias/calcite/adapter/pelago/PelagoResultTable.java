package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.types.PelagoTypeParser;
import com.google.common.collect.Lists;
import org.apache.calcite.DataContext;
import org.apache.calcite.adapter.csv.CsvEnumerator;
import org.apache.calcite.adapter.csv.CsvTableScan;
import org.apache.calcite.linq4j.*;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.*;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Source;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTable.java
 */
public class PelagoResultTable extends AbstractTable implements ScannableTable {//QueryableTable, TranslatableTable {
    protected final RelProtoDataType    protoRowType;
    protected final Source              source      ;
    protected RelDataType               rowType     ;
    protected Map<String, ?>            type        ;
    protected Map<String, ?>            plugin      ;
    protected Long                      linehint    ;
    protected final boolean             mock        ;

    PelagoResultTable(Source source, RelDataType rowType, boolean mock) {
        this.source         = source    ;
        this.type           = null      ;
        this.rowType        = rowType   ;
        this.linehint       = null      ;
        this.protoRowType   = null      ;
        this.mock           = mock      ;
    }

    PelagoResultTable(Source source, Map<String, ?> type, Map<String, ?> plugin, long linehint, boolean mock) {
        this.source     = source    ;
        this.type       = type      ;
        this.rowType    = null      ;
        this.linehint   = linehint  ;
        this.plugin     = plugin    ;

        this.protoRowType = null;
        this.mock       = mock      ;
    }

    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        if (protoRowType != null) return protoRowType.apply(typeFactory);
        try {
            return PelagoTypeParser.parseType(typeFactory, type);
        } catch (IOException e) {
            return null;
        }
    }

    public Statistic getStatistic() {
        if (linehint == null) return Statistics.UNKNOWN;
        double rc = linehint;
        final List<ImmutableBitSet> keys = Lists.newArrayList();
//	  final Content content = supplier.get();
//	  for (Ord<Column> ord : Ord.zip(content.columns)) {
//	    if (ord.e.cardinality == content.size) {
//	      keys.add(ImmutableBitSet.of(ord.i));
//	    }
//	  }
//        keys.add(ImmutableBitSet.of(0));
        return Statistics.of(rc, keys);
    }

    /** Returns an array of integers {0, ..., n - 1}. */
    private static int[] identityList(int n) {
        int[] ints = new int[n];
        for (int i = 0; i < n; i++) ints[i] = i;
        return ints;
    }

    public String getPelagoRelName(){
        return source.path();
    }

    public Map<String, ?> getPluginInfo(){
        return plugin;
    }

    public Long getLineHint(){
        return linehint;
    }

    public Enumerable<Object[]> scan(DataContext root) {
        final int[] fields = PelagoEnumerator.identityList(rowType.getFieldCount());
        final AtomicBoolean cancelFlag = DataContext.Variable.CANCEL_FLAG.get(root);
        return new AbstractEnumerable<Object[]>() {
            public Enumerator<Object[]> enumerator() {
                return new PelagoEnumerator<>(source, cancelFlag, rowType.getFieldList(), fields, mock);
//                return new PelagoEnumerator<>(source, cancelFlag, false, null,
//                        new PelagoEnumerator.ArrayRowConverter(rowType.getFieldList(), fields, mock));
            }
        };
    }

    public Expression getExpression(SchemaPlus schema, String tableName,
                                    Class clazz) {
        return Schemas.tableExpression(schema, getElementType(), tableName, clazz);
    }

    public Type getElementType() {
        return Object[].class;
    }

    public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
                                        SchemaPlus schema, String tableName) {
        throw new UnsupportedOperationException();
    }

    public RelNode toRel(
            RelOptTable.ToRelContext context,
            RelOptTable relOptTable) {
        // Request all fields.
        final int fieldCount = relOptTable.getRowType().getFieldCount();
        final int[] fields = CsvEnumerator.identityList(fieldCount);
        return new PelagoResultScan(context.getCluster(), relOptTable, this, fields);
    }
}
