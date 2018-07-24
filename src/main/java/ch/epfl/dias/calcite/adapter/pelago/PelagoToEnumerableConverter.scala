package ch.epfl.dias.calcite.adapter.pelago

import com.google.common.base.Supplier
import com.google.common.collect.ImmutableList
import org.apache.calcite.DataContext
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.config.CalciteConnectionConfig
import org.apache.calcite.interpreter.Source
import org.apache.calcite.linq4j._
import org.apache.calcite.linq4j.tree._
import org.apache.calcite.materialize.MaterializationService
import org.apache.calcite.plan._
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelDistribution
import org.apache.calcite.rel.RelDistributionTraitDef
import org.apache.calcite.rel.RelDistributions
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.`type`._
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexProgram
import org.apache.calcite.schema._
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.util.BuiltInMethod
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.emitter.{Binding, PlanToJSON}
import ch.epfl.dias.repl.Repl
import com.google.common.base.Function
import com.google.common.collect.Lists
import org.apache.calcite.util.Sources
import java.io.{File, PrintWriter}
import java.util

import ch.epfl.dias.calcite.adapter.pelago.PelagoToEnumerableConverter.process
import ch.epfl.dias.calcite.adapter.pelago.reporting.TimeKeeper
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.avatica.remote.AvaticaHttpClientFactoryImpl
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.{JsonMethods, Serialization}

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

    val plan = getPlan
//    System.out.println(pretty(render(plan)))

    new PrintWriter(Repl.planfile) { write(pretty(render(plan))); close }

    if (mock == true) {
      PelagoToEnumerableConverter.pt = new PelagoResultTable(Sources.of(new File(Repl.mockfile)), getRowType, mock) //TODO: fix path
    } else {
      PelagoToEnumerableConverter.rowType = getRowType
//      val builder = new ProcessBuilder("./rawmain-server")
//      val process = builder.start()
//
//      val stdinWriter = new java.io.PrintWriter((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true);
//      val stdoutReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))
//      val stderrReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getErrorStream()))
//
//      var line: String = "";
//      while ({line = stdoutReader.readLine(); line != null} && line != "ready") {}
//
//      stdinWriter.println("echo results on");
//      stdinWriter.println("execute plan from file plan.json");
//
//
//      while ({line = stdoutReader.readLine(); line != null} && !line.startsWith("result in file")) {
//        System.out.println("pelago: " + line);
//      }
//
//      System.out.println(line)
//      if (line == null){
//        while ({line = stderrReader.readLine(); line != null}) {
//          System.out.println("pelago:err: " + line);
//        }
//      }
//
//      val path = line.substring("result in file ".length);
//      System.out.println("path: " + path);
//
//      PelagoToEnumerableConverter.pt = new PelagoResultTable(Sources.of(new File(path)), getRowType, false)
//
//
//      input.foreach(stdinWriter.write(_))
//      stdinWriter.close()
    }

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
    val traitSet = input.getTraitSet.replace(EnumerableConvention.INSTANCE).replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier[RelDistribution]() {
      override def get: RelDistribution = cluster.getMetadataQuery.distribution(input)
    }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType]() {
      override def get: RelDeviceType = cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)
    })
    new PelagoToEnumerableConverter(input.getCluster, traitSet, input)
  }

  //    private static RelProtoDataType rproto;
  private var pt      : PelagoResultTable = null
  private var rowType : RelDataType       = null

  var builder = if(Repl.isMockRun) null else new ProcessBuilder("./rawmain-server")
  var process = if(Repl.isMockRun) null else builder.start()

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
        builder = new ProcessBuilder("./rawmain-server")
        process = builder.start()

        stdinWriter  = new java.io.PrintWriter((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true)
        stdoutReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))
        stderrReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getErrorStream()))
      }

      if (Repl.echoResults) stdinWriter.println("echo results on" )
      else                  stdinWriter.println("echo results off")

      stdinWriter.println("execute plan from file " + Repl.planfile)

      val tk = TimeKeeper.getInstance()
      var first_texec = true

      while ({line = stdoutReader.readLine(); line != null} && !line.startsWith("result in file")) {
        System.out.println("pelago: " + line)

        if(line.contains("Texecute")){
          val texec = java.lang.Long.parseLong("(\\d+)".r.findFirstIn(line).get)
          if(first_texec) {
            tk.addTexec(texec)
            first_texec = false
          }

        }

        if(line.contains("Tcodegen")){
          val tcodegen = java.lang.Long.parseLong("(\\d+)".r.findFirstIn(line).get)
          tk.addTcodegen(tcodegen);
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

      // print the times
      System.out.println(tk);
      // refresh the times
      tk.refreshTable();

      new PelagoResultTable(Sources.of(new File(path)), rowType, false).scan(root)
    }
  }
//
//  /** E.g. {@code constantArrayList("x", "y")} returns
//    * "Arrays.asList('x', 'y')". */
//  private def constantArrayList[T](values: util.List[T], clazz: Class[_]) = Expressions.call(BuiltInMethod.ARRAYS_AS_LIST.method, Expressions.newArrayInit(clazz, constantList(values)))
//
//  /** E.g. {@code constantList("x", "y")} returns
//    * {@code {ConstantExpression("x"), ConstantExpression("y")}}. */
//  private def constantList[T](values: util.List[T]) = Lists.transform(values, new Function[T, Expression]() {
//    override def apply(a0: T): Expression = return Expressions.constant(a0)
//  })
}
