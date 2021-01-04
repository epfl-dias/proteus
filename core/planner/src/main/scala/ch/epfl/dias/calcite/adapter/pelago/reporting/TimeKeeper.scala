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

package ch.epfl.dias.calcite.adapter.pelago.reporting

import ch.epfl.dias.repl.Repl
import java.sql.Timestamp

/** Global singleton for time measurements */
object TimeKeeper { var INSTANCE = new TimeKeeper }
class TimeKeeper private () {
  private var tPlanToJson: java.lang.Long = 0L
  private var tPlanning: java.lang.Long = 0L
  private var tExecutor: java.lang.Long = 0L
  private var tCodegen: java.lang.Long = 0L
  private var tDataLoad: java.lang.Long = 0L
  private var tCodeOpt: java.lang.Long = 0L
  private var tCodeOptAndLoad: java.lang.Long = 0L
  private var tExec: java.lang.Long = 0L
  private var lastTimestamp: Timestamp = null

  private def reset(): Unit = TimeKeeper.INSTANCE = new TimeKeeper
  def addTexec(time_ms: Long): Unit = tExec = time_ms
  def addTcodegen(time_ms: Long): Unit = tCodegen = time_ms
  def addTdataload(time_ms: Long): Unit = tDataLoad = time_ms
  def addTcodeopt(time_ms: Long): Unit = tCodeOpt = time_ms
  def addTcodeoptnload(time_ms: Long): Unit = tCodeOptAndLoad = time_ms
  def addTplan2json(time_ms: Long): Unit = tPlanToJson = time_ms
  def addTexecutorTime(interval: PelagoTimeInterval): Unit =
    tExecutor = interval.getDifferenceMilli
  def addTplanning(interval: PelagoTimeInterval): Unit =
    tPlanning = interval.getDifferenceMilli
  def addTimestamp(): Unit =
    lastTimestamp = new Timestamp(System.currentTimeMillis)
  def refreshTable(
      query: String,
      cmd_type: String,
      hwmode: String,
      plan: String,
      query_name: String
  ): Unit = {
    if (lastTimestamp == null) addTimestamp()
    TimeKeeperTable.addTimings(
      tExecutor + tPlanning + tPlanToJson,
      tPlanning,
      tPlanToJson,
      tExecutor,
      tCodegen,
      tDataLoad,
      tCodeOpt,
      tCodeOptAndLoad,
      tExec,
      lastTimestamp,
      query,
      cmd_type,
      hwmode,
      plan,
      query_name
    )
    reset()
  }

  override def toString: String = {
    val format: java.lang.String = {
      if (Repl.timingscsv) "Timings,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s"
      else
        "Total time: %dms, " + "Planning time: %dms, " + "Flush plan time: %dms, " + "Total executor time: %dms, " + "Codegen time: %dms, " + "Data Load time: %dms, " + "Code opt time: %dms, " + "Code opt'n'load time: %dms, " + "Execution time: %dms"
    }
    String.format(
      format,
      (tExecutor + tPlanning + tPlanToJson).asInstanceOf[java.lang.Long],
      tPlanning,
      tPlanToJson,
      tExecutor,
      tCodegen,
      tDataLoad,
      tCodeOpt,
      tCodeOptAndLoad,
      tExec,
      lastTimestamp
    )
  }
}
