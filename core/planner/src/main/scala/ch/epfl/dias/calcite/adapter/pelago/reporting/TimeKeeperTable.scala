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

import org.apache.calcite.jdbc.JavaTypeFactoryImpl
import org.apache.calcite.rel.`type`.{RelDataTypeImpl, RelProtoDataType}
import org.apache.calcite.server.PelagoMutableArrayTable
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.sql2rel.{
  InitializerExpressionFactory,
  NullInitializerExpressionFactory
}

import java.sql.Timestamp
import java.util

object TimeKeeperTable {
  var INSTANCE: TimeKeeperTable = init

  private def init = {
    val typeFactory = new JavaTypeFactoryImpl
    val sb = typeFactory.builder
    sb.add("total_time", SqlTypeName.BIGINT)
    sb.add("planning_time", SqlTypeName.BIGINT)
    sb.add("plan2json_time", SqlTypeName.BIGINT)
    sb.add("executor_time", SqlTypeName.BIGINT)
    sb.add("codegen_time", SqlTypeName.BIGINT)
    sb.add("dataload_time", SqlTypeName.BIGINT)
    sb.add("code_opt_time", SqlTypeName.BIGINT)
    sb.add("code_optnload_time", SqlTypeName.BIGINT)
    sb.add("execution_time", SqlTypeName.BIGINT)
    sb.add("timestamp", SqlTypeName.VARCHAR)
    sb.add("query", SqlTypeName.VARCHAR)
    sb.add("cmd_type", SqlTypeName.VARCHAR)
    sb.add("hwmode", SqlTypeName.VARCHAR)
    sb.add("plan", SqlTypeName.VARCHAR)
    sb.add("query_name", SqlTypeName.VARCHAR)
    val ief = new NullInitializerExpressionFactory
    new TimeKeeperTable(
      "Timings",
      RelDataTypeImpl.proto(sb.build),
      RelDataTypeImpl.proto(sb.build),
      ief
    )
  }

  def addTimings(
      ttotal_ms: Long,
      tplanning_ms: Long,
      tplan2json_ms: Long,
      texecutor_ms: Long,
      tcodegen_ms: Long,
      tdataload_ms: Long,
      tcode_opt_time_ms: Long,
      tcode_optnload_time_ms: Long,
      texecution_ms: Long,
      timestamp: Timestamp,
      query: String,
      cmd_type: String,
      hwmode: String,
      plan: String,
      query_name: String
  ): Unit = {
    val arr = Array(
      ttotal_ms,
      tplanning_ms,
      tplan2json_ms,
      texecutor_ms,
      tcodegen_ms,
      tdataload_ms,
      tcode_opt_time_ms,
      tcode_optnload_time_ms,
      texecution_ms,
      timestamp.toString,
      query,
      cmd_type,
      hwmode,
      plan,
      query_name
    )
    INSTANCE.getModifiableCollection
      .asInstanceOf[util.Collection[Array[Any]]]
      .add(arr)
  }
}

class TimeKeeperTable private (
    val name: String,
    val protoStoredRowType: RelProtoDataType,
    val protoRowType: RelProtoDataType,
    val initializerExpressionFactory: InitializerExpressionFactory
) extends PelagoMutableArrayTable(
      name,
      protoStoredRowType,
      protoRowType,
      initializerExpressionFactory
    ) {}
