package ch.epfl.dias.repl

import ch.epfl.dias.calcite.adapter.pelago.executor.PelagoExecutor
import org.apache.calcite.avatica.jdbc.JdbcMeta
import org.apache.calcite.avatica.remote.Driver.Serialization
import org.apache.calcite.avatica.remote.LocalService
import org.apache.calcite.avatica.server.HttpServer

import scala.annotation.tailrec
import scala.sys.process.Process

object Repl extends App {
  Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver")

  val arglist = args.toList
  val defaultMock =
    new java.io.File(".").getCanonicalPath + "/src/main/resources/mock.csv"
  val defaultSchema =
    new java.io.File(".").getCanonicalPath + "/src/main/resources/schema.json"

  type OptionMap = Map[Symbol, Any]

  @tailrec
  def nextOption(map: OptionMap, list: List[String]): OptionMap = {
    list match {
      case Nil => map
      case "--echo-results" :: tail =>
        nextOption(map ++ Map('echoResults -> true), tail)
      case "--port" :: value :: tail =>
        nextOption(map ++ Map('port -> value.toInt), tail)
      case "--cpuonly" :: tail =>
        nextOption(map ++ Map('cpuonly -> true), tail)
      case "--onlyinit" :: tail =>
        nextOption(map ++ Map('onlyinit -> true), tail)
      case "--timings-csv" :: tail =>
        nextOption(map ++ Map('timingscsv -> true), tail)
      case "--mockfile" :: value :: tail =>
        nextOption(map ++ Map('mockfile -> value) ++ Map('mock -> true), tail)
      case "--planfile" :: value :: tail =>
        nextOption(map ++ Map('planfile -> value), tail)
      case "--gpudop" :: value :: tail =>
        nextOption(map ++ Map('gpudop -> value.toInt), tail)
      case "--cpudop" :: value :: tail =>
        nextOption(map ++ Map('cpudop -> value.toInt), tail)
      case "--mock" :: tail =>
        nextOption(map ++ Map('mock -> true), tail)
      case string :: Nil => nextOption(map ++ Map('schema -> string), list.tail)
      case option :: _ =>
        println("Unknown option " + option)
        println(
          "Usage: [--port] [--echo-results] [--timings-csv] [--planfile <path-to-write-plan>] [--mockfile <path-to-mock-file>|--mock] [path-to-schema.json]"
        )
        System.exit(1)
        null
    }
  }

  var executor_server = "./proteuscli-server"

  val topology = Process(executor_server + " --query-topology").!!
  val gpudop_regex = """gpu {2}count: (\d+)""".r.unanchored
  val detected_gpudop = topology match {
    case gpudop_regex(g) => g.toInt
    case _               => 0
  }
  val cpudop_regex = """core count: (\d+)""".r.unanchored
  val detected_cpudop = topology match {
    case cpudop_regex(c) =>
      c.toInt / 2 // We want by default to ignore hyperthreads
    case _ => 48
  }

  System.out.println(topology)

  val options = nextOption(
    Map(
      'port -> 8081,
      'cpudop -> detected_cpudop,
      'gpudop -> detected_gpudop,
      'echoResults -> false,
      'mock -> false,
      'timings -> true,
      'timingscsv -> false,
      'mockfile -> defaultMock,
      'cpuonly -> (detected_gpudop <= 0),
      'planfile -> "plan.json",
      'schema -> defaultSchema,
      'onlyinit -> false,
      'printplan -> false
    ),
    arglist
  )

  System.out.println(options)

  var mockfile = options('mockfile).asInstanceOf[String]
  var isMockRun = options('mock).asInstanceOf[Boolean]
  var echoResults = options('echoResults).asInstanceOf[Boolean]
  var planfile = options('planfile).asInstanceOf[String]
  var printplan = options('printplan).asInstanceOf[Boolean]
  var timingscsv = options('timingscsv).asInstanceOf[Boolean]
  var timings = options('timings).asInstanceOf[Boolean]

  var cpus_on = true //options('cpuonly    ).asInstanceOf[Boolean]
  var gpus_on = true
  var cpudop = options('cpudop).asInstanceOf[Int]
  var gpudop = options('gpudop).asInstanceOf[Int]

  //default to hybrid execution
  setHybrid()
  //check if the user requested cpuonly execution (or no GPUs are available)
  if (options('cpuonly).asInstanceOf[Boolean]) setCpuonly()

  def isHybrid = cpus_on && gpus_on
  def setHybrid(): Unit = {
    cpus_on = true
    gpus_on = true
  }

  def isCpuonly = cpus_on && !gpus_on
  def setCpuonly(): Unit = {
    cpus_on = true
    gpus_on = false
  }

  def isGpuonly = !cpus_on && gpus_on
  def setGpuonly(): Unit = {
    cpus_on = false
    gpus_on = true
  }

  // Not the cleanest way to provide this path, but sbt crashes otherwise. Incompatible with assembly jar
  val schemaPath: String = options('schema).asInstanceOf[String]

  if (options('onlyinit).asInstanceOf[Boolean]) {
    // Done, return from object initialization
  } else {
    val meta = new JdbcMeta("jdbc:proteus:model=" + schemaPath)
    val service = new LocalService(meta)
    val server = new HttpServer.Builder()
      .withHandler(service, Serialization.PROTOBUF)
      .withPort(options('port).asInstanceOf[Int])
      .build()

    PelagoExecutor.restartEngine()
    server.start()
    server.join()
  }
}
