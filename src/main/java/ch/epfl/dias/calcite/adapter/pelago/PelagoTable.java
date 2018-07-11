package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.types.PelagoTypeParser;
import com.google.common.collect.Lists;
import org.apache.calcite.DataContext;
import org.apache.calcite.linq4j.*;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelCollation;
//import org.apache.calcite.rel.RelDeviceType;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
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
public class PelagoTable extends AbstractTable implements TranslatableTable {
    protected final RelProtoDataType    protoRowType;
    protected final Source              source      ;
//    protected RelDataType               rowType     ;
    protected Map<String, ?>            type        ;
    protected Map<String, ?>            plugin      ;
    protected Long                      linehint    ;

    private PelagoTable(Source source, RelProtoDataType protoRowType, Map<String, ?> plugin, long linehint) {
        this.source         = source    ;
        this.type           = null      ;
//        this.rowType        = null      ;
        this.linehint       = linehint  ;
        this.plugin         = plugin    ;

        this.protoRowType   = protoRowType;
    }

    private PelagoTable(Source source, Map<String, ?> type, Map<String, ?> plugin, long linehint) {
        this.source     = source    ;
        this.type       = type      ;
//        this.rowType    = null      ;
        this.linehint   = linehint  ;
        this.plugin     = plugin    ;

        this.protoRowType = null;
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

    public RelNode toRel(
            RelOptTable.ToRelContext context,
            RelOptTable relOptTable) {
        // Request all fields.
//        context.getCluster().getPlanner().addRelTraitDef(RelDistributionTraitDef.INSTANCE);
        final int fieldCount = relOptTable.getRowType().getFieldCount();
        final int[] fields = identityList(fieldCount);
        return PelagoTableScan.create(context.getCluster(), relOptTable, this, fields);
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

    public RelDeviceType   getDeviceType(){
        return RelDeviceType.X86_64;
    }

    public RelDistribution getDistribution(){
        return RelDistributions.SINGLETON;
    }


    private static Long getLineHintFromPlugin(String name, Map<String, ?> plugin) throws MalformedPlugin {
        Object obj_linehint = plugin.getOrDefault("lines",  null);
        if (obj_linehint == null){
            obj_linehint = plugin.getOrDefault("linehint",  null);
        }

        Long linehint = null;
        if (obj_linehint != null) {
            if (obj_linehint instanceof Integer) {
                linehint = ((Integer) obj_linehint).longValue();
            } else if (obj_linehint instanceof Long){
                linehint = (Long) obj_linehint;
            } else {
                throw new MalformedPlugin("\"lines\" unrecognized type for \"lines\" during creation of " + name, name);
            }
        }

        if (linehint == null) {
            throw new MalformedPlugin("\"lines\" not found for table " + name, name);
        }

        return linehint;
    }

    public static PelagoTable create(Source source, String name, Map<String, ?> plugin, Map<String, ?> lineType  ) throws MalformedPlugin {
        return new PelagoTable(source, lineType, plugin, getLineHintFromPlugin(name, plugin));
    }

    public static PelagoTable create(Source source, String name, Map<String, ?> plugin, RelProtoDataType lineType) throws MalformedPlugin {
        return new PelagoTable(source, lineType, plugin, getLineHintFromPlugin(name, plugin));
    }

    public RelPacking getPacking() {
        if (plugin.get("type").toString().equalsIgnoreCase("block")) return RelPacking.Packed;
        return RelPacking.UnPckd;
    }
}
