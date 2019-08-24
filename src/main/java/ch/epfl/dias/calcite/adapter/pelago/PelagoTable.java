package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.types.PelagoTypeParser;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.plan.RelOptTable;
//import org.apache.calcite.rel.RelDeviceType;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.rel.RelReferentialConstraintImpl;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.schema.*;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Source;
import org.apache.calcite.util.mapping.IntPair;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTable.java
 */
public class PelagoTable extends AbstractTable implements TranslatableTable {
    protected final RelProtoDataType    protoRowType;
    protected final RelDataType         rowType;
    protected final Source              source      ;
    protected final String              name        ;
//    protected RelDataType               rowType     ;
    protected Map<String, ?>            type        ;
    protected Map<String, ?>            plugin      ;
    protected Long                      linehint    ;
    protected List<Map<String, ?>>      constraints ;

    protected Map<String, Table>        knownTables ;

    protected Statistic                             stats = null;
    protected ImmutableMap<ImmutableBitSet, Double> dCnt  = null;
    protected ImmutableMap<ImmutableBitSet, Pair<Object, Object>> ranges  = null;
    // This map IS mutable, it should be updated periodically
//    protected Map<ImmutableBitSet, > dCnt  = null;

    private PelagoTable(Source source, RelProtoDataType protoRowType, Map<String, ?> plugin, long linehint, List<Map<String, ?>> constraints) {
        this.source         = source    ;
        this.type           = null      ;
        this.rowType        = null      ;
        this.linehint       = linehint  ;
        this.plugin         = plugin    ;
        this.name           = source.path();

        this.protoRowType   = protoRowType;

        if (constraints != null) {
            this.constraints = constraints;
        } else {
            this.constraints = new ArrayList<>();
        }
    }

    private PelagoTable(String name, RelDataType rowType, List<Map<String, ?>> constraints) {
        this.source         = null;
        this.name           = name;
        this.type           = null      ;
        this.linehint       = Long.MAX_VALUE;
        this.plugin         = Map.of("type", "intermediate");

        this.protoRowType   = null;
        this.rowType        = rowType;

        if (constraints != null) {
            this.constraints = constraints;
        } else {
            this.constraints = new ArrayList<>();
        }
    }

    private PelagoTable(Source source, Map<String, ?> type, Map<String, ?> plugin, long linehint, List<Map<String, ?>> constraints) {
        this.source     = source    ;
        this.type       = type      ;
        this.rowType    = null      ;
        this.linehint   = linehint  ;
        this.plugin     = plugin    ;
        this.name       = source.path();

        this.protoRowType = null;

        if (constraints != null) {
            this.constraints = constraints;
        } else {
            this.constraints = new ArrayList<>();
        }
    }

    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        if (rowType != null && typeFactory == null) return rowType;
        if (protoRowType == null && typeFactory == null) typeFactory = new JavaTypeFactoryImpl();

        if (protoRowType != null) return protoRowType.apply(typeFactory);

        try {
            return PelagoTypeParser.parseType(typeFactory, type);
        } catch (IOException e) {
            return null;
        }
    }

    private int getColumnIndex(String col){
        return getRowType(null).getField(col, false, true).getIndex();
    }

    public void overwriteKnownTables(Map<String, Table> t){
        knownTables = t;
    }

    protected void initStatistics() {
        double rc = linehint;
        final List<ImmutableBitSet> keys = Lists.newArrayList();
        ImmutableMap.Builder dCntBuilder = ImmutableMap.builder();
        ImmutableMap.Builder rangesBuilder = ImmutableMap.builder();
//	  final Content content = supplier.get();
//	  for (Ord<Column> ord : Ord.zip(content.columns)) {
//	    if (ord.e.cardinality == content.size) {
//	      keys.add(ImmutableBitSet.of(ord.i));
//	    }
//	  }
//        keys.add(ImmutableBitSet.of(0));

        ImmutableList.Builder<RelReferentialConstraint> constr = ImmutableList.builder();

        for (Map<String, ?> c: constraints){
            String type = ((String) c.get("type")).toLowerCase();
            switch (type) {
            case "primary_key":
            case "unique": {
                List<String> columns = ((List<String>) c.get("columns"));
                assert (columns.size() > 0);

                ImmutableBitSet.Builder k = ImmutableBitSet.builder();
                for (String col : columns) {
                    k.set(getColumnIndex(col));
                }
                keys.add(k.build());
                break;
            }
            case "distinct_cnt": {
                List<String> columns = ((List<String>) c.get("columns"));
                assert (columns.size() > 0);

                ImmutableBitSet.Builder k = ImmutableBitSet.builder();
                for (String col : columns) {
                    k.set(getColumnIndex(col));
                }
                dCntBuilder.put(k.build(), ((Number) c.get("values")).doubleValue());
                break;
            }
            case "range": {
                String column = ((String) c.get("column"));

                int index = getColumnIndex(column);
                ImmutableBitSet col = ImmutableBitSet.of(index);

                Object litmin = c.getOrDefault("min", null);
                Object litmax = c.getOrDefault("max", null);

                rangesBuilder.put(col, Pair.of(litmin, litmax));
                break;
            }
            case "foreign_key": {
                List<String> columns = ((List<String>) c.get("columns"));
                String tableName = knownTables.entrySet().stream().
                    filter(x -> x.getValue() == this).findAny().get().getKey();

                String ref = ((String) c.get("referencedTable"));

                ImmutableList.Builder<IntPair> refs = ImmutableList.builder();

                List<Map<String, String>> pairs = (List<Map<String, String>>) c.get("references");

                for (Map<String, String> p: pairs){
                    refs.add(IntPair.of(
                        getColumnIndex(p.get("referee")),
                        ((PelagoTable) knownTables.get(ref)).getColumnIndex(p.get("referred"))
                    ));
                }

                constr.add(
                    RelReferentialConstraintImpl.of(
                        ImmutableList.of("SSB", tableName),
                        ImmutableList.of("SSB", ref),
                        refs.build()
                    )
                );
                break;
            }
            default: {
                System.err.println("Unknown statistic: " + type);
                break;
            }
            }
        }

        dCnt = dCntBuilder.build();
        ranges = rangesBuilder.build();
        stats = Statistics.of(rc, keys, constr.build(), ImmutableList.of());
    }


    public Statistic getStatistic() {
        if (stats == null) initStatistics();
        return stats;
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
        RelNode scan = PelagoTableScan.create(context.getCluster(), relOptTable, this, fields);
        if (getPacking() == RelPacking.Packed) scan = PelagoUnpack.create(scan, RelPacking.UnPckd);
        return scan;
    }

    public String getPelagoRelName(){
        return name;
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

    public RelHomDistribution getHomDistribution(){
        return RelHomDistribution.SINGLE;
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

    public static PelagoTable create(Source source, String name, Map<String, ?> plugin, Map<String, ?> lineType  , List<Map<String, ?>> constraints) throws MalformedPlugin {
        return new PelagoTable(source, lineType, plugin, getLineHintFromPlugin(name, plugin), constraints);
    }

    public static PelagoTable create(Source source, String name, Map<String, ?> plugin, RelProtoDataType lineType) throws MalformedPlugin {
        return new PelagoTable(source, lineType, plugin, getLineHintFromPlugin(name, plugin), null);
    }

    public static PelagoTable create(String name, RelDataType lineType) throws MalformedPlugin {
        return new PelagoTable(name, lineType, null);
    }

    public RelPacking getPacking() {
        if (plugin.get("type").toString().equalsIgnoreCase("block")) return RelPacking.Packed;
        return RelPacking.UnPckd;
    }

    public Double getDistrinctValues(ImmutableBitSet cols) {
        if (dCnt == null) initStatistics();
        return dCnt.getOrDefault(cols, null);
    }

    public static BigInteger stringToNum(String x, int chars){
        BigInteger ret = BigInteger.valueOf(0);
        int len = Math.min(chars, x.length());
        for (int i = 0 ; i < len ; ++i) ret = ret.multiply(BigInteger.valueOf(256)).add(BigInteger.valueOf(Math.min(x.charAt(i), 255)));

        ret = ret.multiply(BigInteger.valueOf(256).pow(Math.max(x.length() - len, 0)));

        return ret;
    }

    public static double getPercentile(int min, Double max, Double val){
        return (val - min) / (max - min);
    }

    public static double getPercentile(BigDecimal min, BigDecimal max, BigDecimal val){
        return (val.subtract(min)).doubleValue() / (max.subtract(min)).doubleValue();
    }

    public static double getPercentile(BigInteger min, BigInteger max, BigInteger val){
        return (val.subtract(min)).doubleValue() / (max.subtract(min)).doubleValue();
    }

    public static double getPercentile(String start, String end, String q){
        if (q.compareTo(start) < 0) return 0;
        if (q.compareTo(end  ) > 0) return 1;
        if (end.equals(start)     ) return 1;

        int len = Math.max(Math.max(start.length(), end.length()), q.length());

        BigInteger max = stringToNum(end  , len);
        BigInteger min = stringToNum(start, len);
        BigInteger val = stringToNum(q    , len);

        return getPercentile(min, max, val);
    }

    public Double getPercentile(final ImmutableBitSet col, final RexLiteral val, final RexBuilder rexBuilder) {
        Double dist = getDistrinctValues(col);
        if (dist == null) return null;
        SqlTypeName type = getRowType(null).getFieldList().get(col.nextSetBit(0)).getType().getSqlTypeName();
        if (type == SqlTypeName.CHAR || type == SqlTypeName.VARCHAR || type == SqlTypeName.DECIMAL || type == SqlTypeName.BIGINT || type == SqlTypeName.INTEGER){
            Pair r = ranges.getOrDefault(col, null);
            if (r != null && r.left != null && r.right != null) {
                // While currently we only support VARCHAR literals for ranges, keep the extra step through RexLiteral here to allow for easier generalization
                if (type == SqlTypeName.CHAR || type == SqlTypeName.VARCHAR) {
                    RexLiteral rmin = (RexLiteral) rexBuilder.makeLiteral(r.left , val.getType(), true);
                    RexLiteral rmax = (RexLiteral) rexBuilder.makeLiteral(r.right, val.getType(), true);
                    String min = rmin.getValueAs(String.class);
                    String max = rmax.getValueAs(String.class);
                    return getPercentile(min, max, val.getValueAs(String.class));
                } else {
                    try {
                        BigDecimal min = new BigDecimal(r.left.toString());
                        BigDecimal max = new BigDecimal(r.right.toString());
                        return getPercentile(min, max, val.getValueAs(BigDecimal.class));
                    }catch (Exception e){
                        return null;
                    }
                }
            }
        }
        return RelMdUtil.numDistinctVals(dist, dist * 0.5)/dist;
    }
}
