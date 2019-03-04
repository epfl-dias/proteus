package ch.epfl.dias.calcite.adapter.pelago

import com.google.common.collect.ImmutableList
import org.apache.calcite.DataContext
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.linq4j._
import org.apache.calcite.linq4j.tree._
import org.apache.calcite.plan._
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.{RelDistributionTraitDef, RelNode, RelVisitor, RelWriter}
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.`type`._
import org.apache.calcite.rex.RexInputRef
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.repl.Repl
import org.apache.calcite.util.Sources
import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.file.{Files, Paths}
import java.util

import ch.epfl.dias.calcite.adapter.pelago.reporting.{PelagoTimeInterval, TimeKeeper}
import ch.epfl.dias.emitter.PlanToJSON._
import org.json4s.JValue
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._

/**
  * Relational expression representing a scan of a table in a Pelago data source.
  */

class PelagoToEnumerableConverter private(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode)
  extends ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits, input) with EnumerableRel {

  implicit val formats = DefaultFormats

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): RelNode = copy(traitSet, inputs.get(0))

  def copy(traitSet: RelTraitSet, input: RelNode): RelNode = PelagoToEnumerableConverter.create(input)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    super.computeSelfCost(planner, mq).multiplyBy(getRowType.getFieldCount.toDouble * 0.1)
  }

  def getPlan: JValue = {
    val op = ("operator" , "print")
    val alias = "print" + getId
    val rowType = emitSchema(alias, getRowType)
    val child = getInput.asInstanceOf[PelagoRel].implement(RelDeviceType.X86_64)
    val childBinding: Binding = child._1
    val childOp = child._2

    val exprs = getRowType
    val exprsJS: JValue = exprs.getFieldList.asScala.zipWithIndex.map {
      e => {
        val reg_as = ("attrName", getRowType.getFieldNames.get(e._2)) ~ ("relName", alias)
        emitExpression(RexInputRef.of(e._1.getIndex, getRowType), List(childBinding)).asInstanceOf[JObject] ~ ("register_as", reg_as)
      }
    }

    op ~
      ("gpu"  , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) ~
      ("e"    , exprsJS                                              ) ~
      ("input", childOp                                              ) // ~ ("tupleType", rowType)
  }

  override def implement(implementor: EnumerableRelImplementor, pref: EnumerableRel.Prefer): EnumerableRel.Result = {
    val mock = Repl.isMockRun //TODO: change!!!

    def visit(node: RelNode): Set[String] = {
      if (node.isInstanceOf[PelagoDictTableScan]) {
        return Set() // Otherwise the reduce above has a problem
        // The actual column will be scanned on the other side
      } else if (node.isInstanceOf[PelagoTableScan]){
        val scan = node.asInstanceOf[PelagoTableScan]
        if (scan.pelagoTable.getLineHint <= 1024*1024*1024/8) return Set()
        val relName = scan.pelagoTable.getPelagoRelName
        // FIXME: Should ask the plugin for the list of files
        scan.getRowType.getFieldNames.asScala.map(e => relName + "." + e).toSet
      } else {
        node.getInputs.asScala.map(e => visit(e)).reduce((a, b) => a ++ b)
      }
    }

    PelagoToEnumerableConverter.files = visit(getInput)

    val planTimer = new PelagoTimeInterval
    planTimer.start()

    val plan = getPlan

    new PrintWriter(Repl.planfile) { write(pretty(render(plan))); close }
    if (Files.exists(Paths.get("../../src/panorama/public/assets"))) {new PrintWriter(new FileOutputStream("../../src/panorama/public/assets/flare.json", false)) { write(pretty(render(plan))); close } }

    if (PelagoSplit.bindings.size > 0){
      println(PelagoSplit.bindings)
      throw new RuntimeException("Unmatched split operators (maybe the cost models didn't allow for even push down?)");
    }

    if (mock == true) {
      PelagoToEnumerableConverter.pt = new PelagoResultTable(Sources.of(new File(Repl.mockfile)), getRowType, mock) //TODO: fix path
    } else {
      PelagoToEnumerableConverter.rowType = getRowType
    }

    // report time to create the json and flush it
    planTimer.stop()
    TimeKeeper.INSTANCE.addTplan2json(planTimer.getDifferenceMilli)

    val table = RelOptTableImpl.create(null, getRowType, ImmutableList.of[String](),
      Expressions.call(
        Types.lookupMethod(
          classOf[PelagoToEnumerableConverter],
          "getEnumerableResult",
          classOf[DataContext]
        ),
        DataContext.ROOT
      )
    )
    val fields = new Array[Int](getRowType.getFieldCount)
    for (i <- 0 to fields.length - 1) fields(i) = i

    val ts = new PelagoResultScan(getCluster, table, PelagoToEnumerableConverter.pt, fields)
    ts.implement(implementor, pref)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)
}

object PelagoToEnumerableConverter {
  def create(input: RelNode): RelNode = {
    val cluster = input.getCluster
    val traitSet = input.getTraitSet.replace(EnumerableConvention.INSTANCE)
      .replace(cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].deviceType(input))
    new PelagoToEnumerableConverter(input.getCluster, traitSet, input)
  }

  // FIXME: This variables may be unsafe, what happens with prepared statements?
  private var pt          : PelagoResultTable = null
  private var rowType     : RelDataType       = null
  private var files       : Set[String]       = null
  private var loadedfiles : Set[String]       = null

  var builder = if(Repl.isMockRun) null else new ProcessBuilder(Repl.executor_server)
  var process = if(Repl.isMockRun || builder == null) null else builder.start()

  var stdinWriter  = if(Repl.isMockRun) null else new java.io.PrintWriter  ((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true)
  var stdoutReader = if(Repl.isMockRun) null else new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))
  var stderrReader = if(Repl.isMockRun) null else new java.io.BufferedReader(new java.io.InputStreamReader(process.getErrorStream()))

  var line: String = ""

  @SuppressWarnings(Array("UnusedDeclaration"))
  def getEnumerableResult(root: DataContext): Enumerable[_] = {
    if (rowType == null) {
      pt.scan(root)
    } else {
      if (process == null || !process.isAlive){
        builder = new ProcessBuilder(Repl.executor_server)
        process = builder.start()
        loadedfiles = null

        stdinWriter  = new java.io.PrintWriter((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true)
        stdoutReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))
        stderrReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getErrorStream()))
      }

      if (Repl.echoResults) stdinWriter.println("echo results on" )
      else                  stdinWriter.println("echo results off")

      val executorTimer = new PelagoTimeInterval
      executorTimer.start()

      // If current files are a subset of loaded files, do not unload!
      if (loadedfiles == null || files == null || !files.subsetOf(loadedfiles)) {
        stdinWriter.println("unloadall")
        loadedfiles = files
      }
      stdinWriter.println("execute plan from file " + Repl.planfile)

      var tdataload: Long = 0
      var tcodeopt: Long = 0
      var tcodeoptnload: Long = 0

      while ({line = stdoutReader.readLine(); line != null} && !line.startsWith("result in file")) {
        System.out.println("pelago: " + line)

        if(line.contains("Texecute w sync: ")){
          val texec = java.lang.Long.parseLong("(\\d+)".r.findFirstIn(line).get)
          TimeKeeper.INSTANCE.addTexec(texec)
        }

        if(line.contains("Tcodegen: ")){
          val tcodegen = java.lang.Long.parseLong("(\\d+)".r.findFirstIn(line).get)
          TimeKeeper.INSTANCE.addTcodegen(tcodegen)
        }

        if(line.startsWith("Topen (") && line.contains("):") && !line.contains(",")){
          val m = "(\\d+)ms$".r.findFirstIn(line).get
          val t = m.slice(0, m.length - 2)
          tdataload = tdataload + java.lang.Long.parseLong(t)
        }

        if(line.contains("Optimization time: ")){
          tcodeopt = tcodeopt + java.lang.Long.parseLong("(\\d+)".r.findFirstIn(line).get)
        }

        if(line.contains(" C: ") || line.contains(" G: ")){
          val m = "(\\d+)ms$".r.findFirstIn(line).get
          val t = m.slice(0, m.length - 2)
          tcodeoptnload = tcodeoptnload + java.lang.Long.parseLong(t)
        }
      }

      if (line == null){
        while ({line = stderrReader.readLine(); line != null}) {
          System.out.println("pelago:err: " + line)
        }
      } else {
        System.out.println("pelago: " + line)
      }

      val path = line.substring("result in file ".length)

      executorTimer.stop()
      TimeKeeper.INSTANCE.addTexecutorTime(executorTimer)
      TimeKeeper.INSTANCE.addTdataload(tdataload)
      TimeKeeper.INSTANCE.addTcodeopt(tcodeopt)
      TimeKeeper.INSTANCE.addTcodeoptnload(tcodeoptnload)
      
      if (Repl.timings) {
        // print the times
        System.out.println(TimeKeeper.INSTANCE)
        // refresh the times
        TimeKeeper.INSTANCE.refreshTable()
      }

      new PelagoResultTable(Sources.of(new File(path)), rowType, false).scan(root)
    }
  }
}
