package ch.epfl.dias.calcite.adapter.pelago.executor

import java.io.File

import ch.epfl.dias.calcite.adapter.pelago.{PelagoResultTable, PelagoToEnumerableConverter}
import ch.epfl.dias.calcite.adapter.pelago.reporting.{PelagoTimeInterval, TimeKeeper}
import ch.epfl.dias.repl.Repl
import org.apache.calcite.DataContext
import org.apache.calcite.linq4j.Enumerable
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.util.Sources

class PelagoExecutor{}

object PelagoExecutor {
  // FIXME: This variables may be unsafe, what happens with prepared statements?
  var pt          : PelagoResultTable = null
  var rowType     : RelDataType       = null
  var files       : Set[String]       = null
  private var loadedfiles : Set[String]       = null

  var builder = if(Repl.isMockRun) null else new ProcessBuilder(Repl.executor_server).redirectErrorStream(true)
  var process = if(Repl.isMockRun || builder == null) null else builder.start()

  var stdinWriter  = if(Repl.isMockRun) null else new java.io.PrintWriter  ((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true)
  var stdoutReader = if(Repl.isMockRun) null else new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))

  var line: String = ""

  @SuppressWarnings(Array("UnusedDeclaration"))
  def getEnumerableResult(root: DataContext, label: String): Enumerable[_] = {
    if (rowType == null) {
      pt.scan(root)
    } else {
      val path = run("execute plan from statement " + label, "execute statement", label, null)

      new PelagoResultTable(Sources.of(new File(path)), rowType, false).scan(root)
    }
  }

  def run(command: String, cmd_type: String, plan: String, query_name: String): String = this.synchronized {
    if (process == null || !process.isAlive){
      builder = new ProcessBuilder(Repl.executor_server).redirectErrorStream(true)
      process = builder.start()
      loadedfiles = null
      PelagoToEnumerableConverter.preparedStatementsCache.clear()

      stdinWriter  = new java.io.PrintWriter((new java.io.OutputStreamWriter(new java.io.BufferedOutputStream(process.getOutputStream()))), true)
      stdoutReader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))
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
    stdinWriter.println(command)//Repl.planfile)

    var tdataload: Long = 0
    var tcodeopt: Long = 0
    var tcodeoptnload: Long = 0

    while ({line = stdoutReader.readLine(); line != null} && !line.startsWith("result in file")
      && !line.startsWith("prepared statement with label ")) {
      System.out.println("pelago: " + line)

      if(line.contains("Texecute w sync: ")){
        val texec = java.lang.Long.parseLong(" (\\d+)ms$".r.findFirstMatchIn(line).get.group(1))
        TimeKeeper.INSTANCE.addTexec(texec)
      }

      if(line.contains("Tcodegen: ")){
        val tcodegen = java.lang.Long.parseLong(" (\\d+)ms$".r.findFirstMatchIn(line).get.group(1))
        TimeKeeper.INSTANCE.addTcodegen(tcodegen)
      }

      if(line.startsWith("Topen (") && line.contains("):") && !line.contains(",")){
        val t = " (\\d+)ms$".r.findFirstMatchIn(line).get.group(1)
        tdataload = tdataload + java.lang.Long.parseLong(t)
      }

      if(line.contains("Optimization time: ")){
        tcodeopt = tcodeopt + java.lang.Long.parseLong(" (\\d+)ms$".r.findFirstMatchIn(line).get.group(1))
      }

      if(line.contains(" C: ") || line.contains(" G: ")){
        val t = " (\\d+)ms$".r.findFirstMatchIn(line).get.group(1)
        tcodeoptnload = tcodeoptnload + java.lang.Long.parseLong(t)
      }
    }

    if (line != null){
      System.out.println("pelago: " + line)
    }

    val path = if (line.startsWith("result in file")) {
      line.substring("result in file ".length)
    } else {
      line.substring("prepared statement with label ".length)
    }

    executorTimer.stop()
    TimeKeeper.INSTANCE.addTexecutorTime(executorTimer)
    TimeKeeper.INSTANCE.addTdataload(tdataload)
    TimeKeeper.INSTANCE.addTcodeopt(tcodeopt)
    TimeKeeper.INSTANCE.addTcodeoptnload(tcodeoptnload)

    if (Repl.timings) {
      // print the times
      System.out.println(TimeKeeper.INSTANCE)
      // refresh the times
      TimeKeeper.INSTANCE.refreshTable(path,
        cmd_type, if (Repl.isHybrid()) "hybrid" else if (Repl.isCpuonly()) "cpuonly" else "gpuonly", plan, query_name)
    }

    path
  }
}
