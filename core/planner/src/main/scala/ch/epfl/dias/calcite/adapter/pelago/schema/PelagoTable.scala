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
package ch.epfl.dias.calcite.adapter.pelago.schema

import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelDeviceType,
  RelHomDistribution,
  RelPacking
}
import ch.epfl.dias.calcite.adapter.pelago.types.PelagoTypeParser
import com.google.common.collect.{ImmutableList, ImmutableMap, Lists}
import org.apache.calcite.jdbc.JavaTypeFactoryImpl
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.`type`.{
  RelDataType,
  RelDataTypeFactory,
  RelProtoDataType
}
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.logical.LogicalTableScan
import org.apache.calcite.rel.metadata.RelMdUtil
import org.apache.calcite.rel.{
  RelCollation,
  RelReferentialConstraint,
  RelReferentialConstraintImpl
}
import org.apache.calcite.rex.{RexBuilder, RexLiteral}
import org.apache.calcite.schema._
import org.apache.calcite.schema.impl.AbstractTable
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.util.mapping.IntPair
import org.apache.calcite.util.{ImmutableBitSet, Pair, Source}

import java.io.IOException
import java.math.{BigDecimal, BigInteger}
import java.util.Objects
import java.{lang, util}
import scala.collection.JavaConverters._

/**
  * Based on:
  * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTable.java
  */
object PelagoTable {
  @throws[MalformedPlugin]
  private def getLineHintFromPlugin(
      name: String,
      plugin: util.Map[String, Any]
  ) = {
    var obj_linehint = plugin.getOrDefault("lines", null)
    if (obj_linehint == null)
      obj_linehint = plugin.getOrDefault("linehint", null)
    var linehint: java.lang.Long = null
    if (obj_linehint != null)
      obj_linehint match {
        case integer: Integer => linehint = integer.longValue
        case l: Long          => linehint = l
        case _ =>
          throw new MalformedPlugin(
            "\"lines\" unrecognized type for \"lines\" during creation of " + name,
            name
          )
      }
    if (linehint == null)
      throw new MalformedPlugin("\"lines\" not found for table " + name, name)
    linehint
  }

  @throws[MalformedPlugin]
  def create(
      source: Source,
      name: String,
      plugin: util.Map[String, Any],
      lineType: util.Map[String, Any],
      constraints: util.List[util.Map[String, Any]]
  ) =
    new PelagoTable(
      source,
      lineType,
      plugin,
      getLineHintFromPlugin(name, plugin),
      constraints,
      name
    )

  @throws[MalformedPlugin]
  def create(
      source: Source,
      name: String,
      plugin: util.Map[String, Any],
      lineType: RelProtoDataType,
      alias: String
  ) =
    new PelagoTable(
      source,
      lineType,
      plugin,
      getLineHintFromPlugin(name, plugin),
      null,
      alias
    )

  @throws[MalformedPlugin]
  def create(
      source: Source,
      name: String,
      plugin: util.Map[String, Any],
      lineType: RelProtoDataType
  ): PelagoTable = create(source, name, plugin, lineType, name)

  @throws[MalformedPlugin]
  def create(name: String, lineType: RelDataType) =
    new PelagoTable(name, lineType, null)

  private def stringToNum(x: String, chars: Int) = {
    var ret = BigInteger.valueOf(0)
    val len = Math.min(chars, x.length)
    for (i <- 0 until len) {
      ret = ret
        .multiply(BigInteger.valueOf(256))
        .add(BigInteger.valueOf(Math.min(x.charAt(i), 255)))
    }
    ret = ret.multiply(BigInteger.valueOf(256).pow(Math.max(x.length - len, 0)))
    ret
  }

  private def getPercentile(min: Int, max: Double, v: Double) =
    (v - min) / (max - min)

  private def getPercentile(min: BigDecimal, max: BigDecimal, v: BigDecimal) =
    v.subtract(min).doubleValue / max.subtract(min).doubleValue

  private def getPercentile(min: BigInteger, max: BigInteger, v: BigInteger) =
    v.subtract(min).doubleValue / max.subtract(min).doubleValue

  def getPercentile(start: String, end: String, q: String): Double = {
    if (q.compareTo(start) < 0) return 0
    if (q.compareTo(end) > 0) return 1
    if (end == start) return 1
    val len = Math.max(Math.max(start.length, end.length), q.length)
    val max = stringToNum(end, len)
    val min = stringToNum(start, len)
    val v = stringToNum(q, len)
    getPercentile(min, max, v)
  }
}

class PelagoTable extends AbstractTable with TranslatableTable {
  final protected var protoRowType: RelProtoDataType = null
  final protected var rowType: RelDataType = null
  final protected var source: Source = null
  final protected var name: String = null
  final var alias: String = null
//    protected RelDataType               rowType     ;
  protected var `type`: util.Map[String, Any] = null
  var plugin: util.Map[String, Any] = null
  protected var linehint = 0L
  protected var constraints: util.List[util.Map[String, Any]] = null
  protected var knownTables: util.Map[String, Table] = null
  protected var stats: Statistic = null
  protected var dCnt: ImmutableMap[ImmutableBitSet, java.lang.Double] = null
  protected var ranges: ImmutableMap[ImmutableBitSet, Pair[Any, Any]] = null
  def this(
      source: Source,
      protoRowType: RelProtoDataType,
      plugin: util.Map[String, Any],
      linehint: Long,
      constraints: util.List[util.Map[String, Any]],
      alias: String
  ) {
    this()
    this.source = source
    this.`type` = null
    this.rowType = null
    this.linehint = linehint
    this.plugin = plugin
    this.name = source.path
    this.alias = alias
    this.protoRowType = protoRowType
    this.constraints = Objects.requireNonNullElseGet(
      constraints,
      () => Lists.newArrayList[util.Map[String, Any]]
    )
  }

  def this(
      name: String,
      rowType: RelDataType,
      constraints: util.List[util.Map[String, Any]]
  ) {
    this()
    this.source = null
    this.name = name
    this.alias = name
    this.`type` = null
    this.linehint = Long.MaxValue
    this.plugin = ImmutableMap.of("type", "intermediate")
    this.protoRowType = null
    this.rowType = rowType
    this.constraints = Objects.requireNonNullElseGet(
      constraints,
      () => Lists.newArrayList[util.Map[String, Any]]
    )
  }

  def this(
      source: Source,
      `type`: util.Map[String, Any],
      plugin: util.Map[String, Any],
      linehint: Long,
      constraints: util.List[util.Map[String, Any]],
      alias: String
  ) {
    this()
    this.source = source
    this.`type` = `type`
    this.rowType = null
    this.linehint = linehint
    this.plugin = plugin
    this.name = source.path
    this.alias = alias
    this.protoRowType = null
    this.constraints = Objects.requireNonNullElseGet(
      constraints,
      () => Lists.newArrayList[util.Map[String, Any]]
    )
  }

  override def getRowType(typeFactory: RelDataTypeFactory): RelDataType = {
    if (rowType != null && typeFactory == null) return rowType
    val tfact =
      if (protoRowType == null && typeFactory == null)
        new JavaTypeFactoryImpl
      else
        typeFactory

    if (protoRowType != null) return protoRowType.apply(tfact)

    try PelagoTypeParser.parseType(tfact, `type`)
    catch {
      case _: IOException =>
        null
    }
  }

  private def getColumnIndex(col: String) =
    getRowType(null).getField(col, false, true).getIndex

  def overwriteKnownTables(t: util.Map[String, Table]): Unit =
    knownTables = t

  private def initStatistics(): Unit = {
    val rc = linehint
    val keys = Lists.newArrayList[ImmutableBitSet]
    val dCntBuilder = ImmutableMap.builder[ImmutableBitSet, java.lang.Double]
    val rangesBuilder = ImmutableMap.builder[ImmutableBitSet, Pair[Any, Any]]
//	  final Content content = supplier.get();
//	  for (Ord<Column> ord : Ord.zip(content.columns)) {
//	    if (ord.e.cardinality == content.size) {
//	      keys.add(ImmutableBitSet.of(ord.i));
//	    }
//	  }
//        keys.add(ImmutableBitSet.of(0));
    val constr = ImmutableList.builder[RelReferentialConstraint]
    for (c <- constraints.asScala) {
      val `type` = c.get("type").asInstanceOf[String].toLowerCase
      `type` match {
        case "primary_key" =>
        case "unique" =>
          val columns = c.get("columns").asInstanceOf[util.List[String]]
          assert(columns.size > 0)
          val k = ImmutableBitSet.builder
          for (col <- columns.asScala) { k.set(getColumnIndex(col)) }
          keys.add(k.build)

        case "distinct_cnt" =>
          val columns = c.get("columns").asInstanceOf[util.List[String]]
          assert(columns.size > 0)
          val k = ImmutableBitSet.builder
          for (col <- columns.asScala) { k.set(getColumnIndex(col)) }
          dCntBuilder.put(
            k.build,
            c.get("values").asInstanceOf[Number].doubleValue
          )

        case "range" =>
          val column = c.get("column").asInstanceOf[String]
          val index = getColumnIndex(column)
          val col = ImmutableBitSet.of(index)
          val litmin = c.getOrDefault("min", null)
          val litmax = c.getOrDefault("max", null)
          rangesBuilder.put(col, Pair.of(litmin, litmax))

        case "foreign_key" =>
          val columns = c.get("columns").asInstanceOf[util.List[String]]
          val tableName = knownTables.entrySet.stream
            .filter((x: util.Map.Entry[String, Table]) => x.getValue eq this)
            .findAny
            .get
            .getKey
          val ref = c.get("referencedTable").asInstanceOf[String]
          val refs = ImmutableList.builder[IntPair]
          val pairs = c
            .get("references")
            .asInstanceOf[util.List[util.Map[String, String]]]

          for (p <- pairs.asScala) {
            refs.add(
              IntPair.of(
                getColumnIndex(p.get("referee")),
                knownTables
                  .get(ref)
                  .asInstanceOf[PelagoTable]
                  .getColumnIndex(p.get("referred"))
              )
            )
          }
          constr.add(
            RelReferentialConstraintImpl.of(
              ImmutableList.of("SSB", tableName),
              ImmutableList.of("SSB", ref),
              refs.build
            )
          )

        case _ =>
          System.err.println("Unknown statistic: " + `type`)

      }
    }
    dCnt = dCntBuilder.build
    ranges = rangesBuilder.build
    stats =
      Statistics.of(rc, keys, constr.build, ImmutableList.of[RelCollation])
  }

  override def getStatistic: Statistic = {
    if (stats == null) initStatistics()
    stats
  }

  override def toRel(
      context: RelOptTable.ToRelContext,
      relOptTable: RelOptTable
  ): LogicalTableScan =
    LogicalTableScan.create(
      context.getCluster,
      relOptTable,
      util.List.of[RelHint]
    )

  def getPelagoRelName: String = name

  def getPluginInfo: util.Map[String, Any] = plugin

  def getLineHint: java.lang.Long = linehint

  def getDeviceType: RelDeviceType = RelDeviceType.X86_64

  def getHomDistribution: RelHomDistribution = RelHomDistribution.SINGLE

  def getPacking: RelPacking = {
    if (plugin.get("type").toString.contains("block")) return RelPacking.Packed
    RelPacking.UnPckd
  }
  def getDistrinctValues(cols: ImmutableBitSet): lang.Double = {
    if (dCnt == null) initStatistics()
    dCnt.getOrDefault(cols, null)
  }

  def getRangeValues(cols: ImmutableBitSet): Pair[Any, Any] = {
    if (ranges == null) initStatistics()
    ranges.getOrDefault(cols, null)
  }

  def getPercentile(
      col: ImmutableBitSet,
      `val`: RexLiteral,
      rexBuilder: RexBuilder
  ): lang.Double = {
    val dist = getDistrinctValues(col)
    if (dist == null) return null
    val `type` = getRowType(null).getFieldList
      .get(col.nextSetBit(0))
      .getType
      .getSqlTypeName
    if (
      (`type` eq SqlTypeName.CHAR) || (`type` eq SqlTypeName.VARCHAR) || (`type` eq SqlTypeName.DECIMAL) || (`type` eq SqlTypeName.BIGINT) || (`type` eq SqlTypeName.INTEGER)
    ) {
      val r = ranges.getOrDefault(col, null)
      if (r != null && r.left != null && r.right != null) { // While currently we only support VARCHAR literals for ranges, keep the extra step through RexLiteral here to allow for easier generalization
        if ((`type` eq SqlTypeName.CHAR) || (`type` eq SqlTypeName.VARCHAR)) {
          val rmin = rexBuilder
            .makeLiteral(r.left, `val`.getType, true)
            .asInstanceOf[RexLiteral]
          val rmax = rexBuilder
            .makeLiteral(r.right, `val`.getType, true)
            .asInstanceOf[RexLiteral]
          val min = rmin.getValueAs(classOf[String])
          val max = rmax.getValueAs(classOf[String])
          return PelagoTable.getPercentile(
            min,
            max,
            `val`.getValueAs(classOf[String])
          )
        } else
          try {
            val min = new BigDecimal(r.left.toString)
            val max = new BigDecimal(r.right.toString)
            return PelagoTable.getPercentile(
              min,
              max,
              `val`.getValueAs(classOf[BigDecimal])
            )
          } catch {
            case e: Exception =>
              return null
          }
      }
    }
    RelMdUtil.numDistinctVals(dist, dist * 0.5) / dist
  }
}
