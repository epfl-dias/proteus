package ch.epfl.dias.calcite.adapter.pelago

import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.file.{Files, Paths}
import java.util

import ch.epfl.dias.calcite.adapter.pelago.executor.PelagoExecutor
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.calcite.adapter.pelago.reporting.{PelagoTimeInterval, TimeKeeper}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import ch.epfl.dias.repl.Repl
import com.google.common.collect.ImmutableList
import org.apache.calcite.DataContext
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.linq4j.tree._
import org.apache.calcite.plan._
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, RelWriter}
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.sql.SqlExplainLevel
import org.apache.calcite.util.Sources
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{JValue, _}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Relational expression representing a scan of a table in a Pelago data source.
  */

class PelagoToEnumerableConverter protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode)
  extends ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits, input) with EnumerableRel {

  implicit val formats: DefaultFormats.type = DefaultFormats

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): RelNode = copy(traitSet, inputs.get(0))

  def copy(traitSet: RelTraitSet, input: RelNode): RelNode = PelagoToEnumerableConverter.create(input)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    super.computeSelfCost(planner, mq).multiplyBy(getRowType.getFieldCount.toDouble * 0.1)
  }

  def getPlan: JValue = {
    val op = ("operator" , "print")
    val alias = PelagoTable.create("print" + getId, getRowType)
    val rowType = emitSchema(alias, getRowType)
    val child = getInput.asInstanceOf[PelagoRel].implement(RelDeviceType.X86_64)
    val childBinding: Binding = child._1
    val childOp = child._2

    val exprs = getRowType
    val exprsJS: JValue = exprs.getFieldList.asScala.zipWithIndex.map {
      e => {
        val reg_as = ("attrName", getRowType.getFieldNames.get(e._2)) ~ ("relName", alias.getPelagoRelName)
        emitExpression(RexInputRef.of(e._1.getIndex, getRowType), List(childBinding), this).asInstanceOf[JObject] ~ ("register_as", reg_as)
      }
    }

    op ~
      ("gpu"  , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) ~
      ("e"    , exprsJS                                              ) ~
      ("input", childOp                                              ) // ~ ("tupleType", rowType)
  }

  def writePlan(plan: JValue, file: String): PrintWriter = {
    new PrintWriter(file) { write(pretty(render(plan))); close() }
  }

  override def implement(implementor: EnumerableRelImplementor, pref: EnumerableRel.Prefer): EnumerableRel.Result = {
    val digest = RelOptUtil.toString(getInput, SqlExplainLevel.DIGEST_ATTRIBUTES)

    val (label, files, rowType) = PelagoToEnumerableConverter.preparedStatementsCache.getOrElseUpdate(
      digest, {
        val mock = Repl.isMockRun //TODO: change!!!

        def visit(node: RelNode): Set[String] = {
          node match {
            case _: PelagoDictTableScan =>
              Set() // Otherwise the reduce above has a problem
            // The actual column will be scanned on the other side
            case scan: PelagoTableScan =>
              if (scan.pelagoTable.getLineHint <= 1024 * 1024 * 1024 / 8) return Set()
              val relName = scan.pelagoTable.getPelagoRelName
              // FIXME: Should ask the plugin for the list of files
              scan.getRowType.getFieldNames.asScala.map(e => relName + "." + e).toSet
            case _: PelagoValues =>
              Set()
            case _ =>
              node.getInputs.asScala.map(e => visit(e)).reduce((a, b) => a ++ b)
          }
        }

        val planTimer = new PelagoTimeInterval
        planTimer.start()

        PelagoSplit.bindings.clear

        val plan = getPlan

        println(Repl.planfile)
        writePlan(plan, Repl.planfile)
        if (Files.exists(Paths.get("../../src/panorama/public/assets"))) {new PrintWriter(new FileOutputStream("../../src/panorama/public/assets/flare.json", false)) { write(pretty(render(plan))); close } }

        if (PelagoSplit.bindings.nonEmpty){
          println(PelagoSplit.bindings)
          throw new RuntimeException("Unmatched split operators (maybe the cost models didn't allow for even push down?)");
        }

        if (mock) {
          PelagoExecutor.pt = new PelagoResultTable(Sources.of(new File(Repl.mockfile)), getRowType, mock) //TODO: fix path
        } else {
          PelagoExecutor.rowType = getRowType
        }

        // report time to create the json and flush it
        planTimer.stop()
        TimeKeeper.INSTANCE.addTplan2json(planTimer.getDifferenceMilli)

        val files = visit(getInput)
        PelagoExecutor.files = files;
        (
          if (mock) "mock" else PelagoExecutor.run("prepare plan from file " + Repl.planfile),
          files,
          getRowType
        )
      }
    )

    PelagoExecutor.rowType = rowType
    PelagoExecutor.files = files

    val table = RelOptTableImpl.create(null, getRowType, ImmutableList.of[String](),
      Expressions.call(
        Types.lookupMethod(
          classOf[PelagoExecutor],
          "getEnumerableResult",
          classOf[DataContext],
          classOf[String]
        ),
        DataContext.ROOT,
        Expressions.constant(label)
      )
    )
    val fields = new Array[Int](getRowType.getFieldCount)
    for (i <- fields.indices) fields(i) = i

    val ts = new PelagoResultScan(getCluster, table, PelagoExecutor.pt, fields)
    ts.implement(implementor, pref)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)
}

object PelagoToEnumerableConverter {
  def create(input: RelNode): RelNode = {
    val cluster = input.getCluster
    val traitSet = input.getTraitSet.replace(EnumerableConvention.INSTANCE)
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].deviceType(input))
    new PelagoToEnumerableConverter(input.getCluster, traitSet, input)
  }

  var preparedStatementsCache: mutable.Map[String, (String, Set[String], RelDataType)] = mutable.Map[String, (String, Set[String], RelDataType)]()
}
