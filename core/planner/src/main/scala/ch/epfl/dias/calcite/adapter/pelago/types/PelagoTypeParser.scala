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

package ch.epfl.dias.calcite.adapter.pelago.types

import org.apache.calcite.linq4j.tree.Primitive
import org.apache.calcite.rel.`type`.{RelDataType, RelDataTypeFactory}
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.util.Pair

import java.io.IOException
import java.util
import scala.collection.JavaConverters._

object PelagoTypeParser {

  @throws[IOException]
  def parseType(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType =
    `type`
      .getOrDefault("type", null)
      .asInstanceOf[String] match {
      case "int" =>
        parseInt(typeFactory, `type`)
      case "int64" =>
        parseInt64(typeFactory, `type`)
      case "float" =>
        parseFloat(typeFactory, `type`)
      case "bool" =>
        parseBoolean(typeFactory, `type`)
      case "dstring" =>
        parseDString(typeFactory, `type`)
      case "date" =>
        parseDate(typeFactory, `type`)
      case "datetime" =>
        parseDatetime(typeFactory, `type`)
      case "string" =>
        parseString(typeFactory, `type`)
      case "set" =>
        parseSet(typeFactory, `type`)
      case "bag" =>
        parseBag(typeFactory, `type`)
      case "list" =>
        parseList(typeFactory, `type`)
      case "record" =>
        parseRecord(typeFactory, `type`)
      case _ =>
        throw new IOException("unknown type: " + `type`)
    }

  def parseInt(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "int")
    val javaType = typeFactory.createJavaType(Primitive.INT.boxClass)
    val sqlType = typeFactory.createSqlType(javaType.getSqlTypeName)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseInt64(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "int64")
    val javaType = typeFactory.createJavaType(Primitive.LONG.boxClass)
    val sqlType = typeFactory.createSqlType(javaType.getSqlTypeName)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseFloat(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "float")
    val javaType = typeFactory.createJavaType(Primitive.DOUBLE.boxClass)
    val sqlType = typeFactory.createSqlType(javaType.getSqlTypeName)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseDate(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "date")
    val sqlType = typeFactory.createSqlType(SqlTypeName.DATE)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseDatetime(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "datetime")
    val sqlType = typeFactory.createSqlType(SqlTypeName.TIMESTAMP)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseBoolean(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "bool")
    val javaType = typeFactory.createJavaType(Primitive.BOOLEAN.boxClass)
    val sqlType = typeFactory.createSqlType(javaType.getSqlTypeName)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseDString(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "dstring")
    val sqlType = typeFactory.createSqlType(SqlTypeName.VARCHAR)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  def parseString(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "string")
    val sqlType = typeFactory.createSqlType(SqlTypeName.VARCHAR)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  @throws[IOException]
  def parseSet(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "set")
    val innerType = parseType(
      typeFactory,
      `type`.get("inner").asInstanceOf[util.Map[String, Any]]
    )
    val sqlType =
      typeFactory.createMultisetType(
        innerType,
        -1
      ) //TODO: not really a multiset
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  @throws[IOException]
  def parseBag(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "bag")
    val innerType = parseType(
      typeFactory,
      `type`.get("inner").asInstanceOf[util.Map[String, Any]]
    )
    val sqlType = typeFactory.createMultisetType(innerType, -1)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  @throws[IOException]
  def parseList(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "list")
    val innerType = parseType(
      typeFactory,
      `type`.get("inner").asInstanceOf[util.Map[String, Any]]
    )
    val sqlType = typeFactory.createArrayType(innerType, -1)
    typeFactory.createTypeWithNullability(sqlType, true)
  }

  @throws[IOException]
  def parseRecord(
      typeFactory: RelDataTypeFactory,
      `type`: util.Map[String, Any]
  ): RelDataType = {
    assert(`type`.getOrDefault("type", null) == "record")
    val attributes =
      `type`.get("attributes").asInstanceOf[util.List[util.Map[String, _]]]
    val types = new util.ArrayList[RelDataType]
    val names = new util.ArrayList[String]
    for (attr <- attributes.asScala) {
      types.add(
        parseType(
          typeFactory,
          attr.get("type").asInstanceOf[util.Map[String, Any]]
        )
      )
      names.add(attr.get("attrName").asInstanceOf[String])
    }
    typeFactory.createStructType(Pair.zip(names, types))
  }
}
