package ch.epfl.dias.emitter

import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.avatica.util.TimeUnitRange
import org.apache.calcite.rel.`type`.{
  RelDataType,
  RelDataTypeField,
  RelRecordType
}
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.{RelNode, SingleRel}
import org.apache.calcite.rex._
import org.apache.calcite.sql.`type`.{SqlTypeName, SqlTypeUtil}
import org.apache.calcite.sql.fun.{
  SqlCaseOperator,
  SqlLikeOperator,
  SqlStdOperatorTable
}
import org.apache.calcite.sql.{pretty => _, _}
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.{PrintWriter, StringWriter}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.control.Breaks._

case class Binding(rel: PelagoTable, fields: List[RelDataTypeField])

object PlanToJSON {
  implicit val formats: Formats = DefaultFormats
  var dictEncoded = false

  //-> TODO What about degree of materialization for joins?
  //---> Join behavior can be handled implicitly by executor:
  //-----> Early materialization: Create new OID
  //-----> Late materialization: Preserve OID + rowTypes of join's children instead of the new one

  def jsonToString(in: JValue): String = {
    pretty(render(in))
  }

  //Code almost verbatim to RexLiteral functionality
  def toJavaString(lit: RexLiteral): String = {
    if (lit.getValue == null) return "null"
    val sw = new StringWriter
    val pw = new PrintWriter(sw)
    lit.printAsJava(pw)
    pw.flush()
    sw.toString
  }

  case class ProjectedRelDataTypeField(actual: RelDataTypeField, orig: String)
      extends RelDataTypeField {
    override def getType: RelDataType = actual.getType
    override def getIndex: Int = actual.getIndex
    override def getName: String = actual.getName
    override def isDynamicStar: Boolean = actual.isDynamicStar
    override def getKey: String = actual.getKey
    override def getValue: RelDataType = actual.getValue
    override def setValue(v: RelDataType): RelDataType = actual.setValue(v)

    def getOrigin: String = orig
  }

  def getFields(
      t: RelDataType,
      origin: String = null
  ): List[ProjectedRelDataTypeField] =
    t match {
      case recType: RelRecordType =>
        recType.getFieldList.asScala
          .map(f => ProjectedRelDataTypeField(f, origin))
          .toList
      case _ =>
        val msg: String = "getFields() assumes RecordType as default input"
        throw new PlanConversionException(msg)
    }

  def emitExpression(e: RexNode, f: List[Binding], input: RelNode): JValue = {
    emitExpression(e, f, arg_with_type = false, input)
  }

  def emitExpression(
      e: RexNode,
      f: List[Binding],
      other: RexNode,
      input: RelNode
  ): JValue = {
    emitExpression(e, f, other, arg_with_type = false, input)
  }

  def emitExpression(
      e: RexNode,
      f: List[Binding],
      arg_with_type: Boolean,
      input: RelNode
  ): JValue = {
    emitExpression(e, f, null, arg_with_type, input)
  }

  def emitLiteral(lit: RexLiteral): JValue =
    lit.getType.getSqlTypeName match {
      case SqlTypeName.INTEGER => new Integer(lit.toString).asInstanceOf[Int]
      case SqlTypeName.BIGINT =>
        new java.lang.Long(lit.toString).asInstanceOf[Long]
      case SqlTypeName.BOOLEAN =>
        new java.lang.Boolean(lit.toString).asInstanceOf[Boolean]
      case SqlTypeName.FLOAT =>
        lit.getValue.asInstanceOf[java.math.BigDecimal].doubleValue()
      case SqlTypeName.DOUBLE =>
        lit.getValue.asInstanceOf[java.math.BigDecimal].doubleValue()
      case SqlTypeName.DECIMAL =>
        lit.getValue.asInstanceOf[java.math.BigDecimal].doubleValue()
      case SqlTypeName.DATE | SqlTypeName.TIMESTAMP =>
        val sw = new StringWriter
        val pw = new PrintWriter(sw)
        lit.printAsJava(pw)
        pw.flush()
        sw.toString
      case SqlTypeName.VARCHAR =>
        lit.getValueAs(classOf[String]) //.toString.substring(1, lit.to)
      case SqlTypeName.CHAR => lit.getValueAs(classOf[String])
      case _ =>
        val msg: String = "Unknown constant type"
        throw new PlanConversionException(msg)
      //List[RelDataTypeField]()
    }

  def emitExpression(
      e: RexNode,
      f: List[Binding],
      other: RexNode,
      arg_with_type: Boolean,
      input: RelNode
  ): JValue = {
    val exprType: JValue = emitType(e.getType, f)
    val json: JValue = e match {
      case call: RexCall =>
        emitOp(call.op, call.operands, exprType, f, input)
      case inputRef: RexInputRef =>
        val arg = emitArg(inputRef, f, arg_with_type)
        arg
      case lit: RexLiteral =>
        if (lit.isNull) {
          ("expression", exprType \ "type") ~ ("isNull", true)
        } else {
          val v = emitLiteral(lit)
          if (
            other != null && lit.getType.getSqlTypeName == SqlTypeName.VARCHAR || lit.getType.getSqlTypeName == SqlTypeName.CHAR
          ) {
            // Only comparisons of input fields with string constants are supported
            // otherwise, which dictionary should we use?

            // Also, input should be a filter operator
            val ref = input.getCluster.getMetadataQuery.getExpressionLineage(
              input.asInstanceOf[SingleRel].getInput(),
              other
            )
            assert(
              ref != null,
              "Have you forgot to add an operator in the expression lineage metadata provider?"
            )
            assert(ref.size == 1)
            // NOTE: ok! that's a good sign!
            val table = ref.iterator.next
              .asInstanceOf[RexTableInputRef]
              .getTableRef
              .getTable

            val info = findAttrInfo(other.asInstanceOf[RexInputRef], f)
            val path = table
              .unwrap(classOf[PelagoTable])
              .getPelagoRelName + "." + info._1.getName + ".dict"

            val old = PlanToJSON.dictEncoded
            PlanToJSON.dictEncoded = table
              .unwrap(classOf[PelagoTable])
              .getPluginInfo
              .get("type")
              .toString
              .contains("block")
            val exprType2: JValue = emitType(e.getType, f)
            PlanToJSON.dictEncoded = old

            ("expression", exprType2 \ "type") ~ ("v", v) ~ ("dict", (
              "path",
              path
            ))
          } else {
            ("expression", exprType \ "type") ~ ("v", v)
          }
        }
      case _ =>
        val msg: String =
          "Unsupported expression " + e.toString + " (" + e.getClass.toString + ")"
        throw new PlanConversionException(msg)
    }
    json
  }

  def aggKind(agg: SqlAggFunction): String =
    agg.getKind match {
      case SqlKind.AVG   => "avg"
      case SqlKind.COUNT => "sum"
      case SqlKind.MAX   => "max"
      case SqlKind.MIN   => "min"
      case SqlKind.SUM   => "sum"
      //'Sum0 is an aggregator which returns the sum of the values which go into it like Sum.'
      //'It differs in that when no non null values are applied zero is returned instead of null.'
      case SqlKind.SUM0    => "sum"
      case SqlKind.COLLECT => "bagunion"
      case _ =>
        val msg: String = "unknown aggr. function " + agg.getKind.toString
        throw new PlanConversionException(msg)
    }

  //TODO Assumming unary ops
  def emitAggExpression(aggExpr: AggregateCall, f: List[Binding]): JValue = {
    val opType: String = aggKind(aggExpr.getAggregation)

    if (aggExpr.getArgList.size() > 1) {
      //count() has 0 arguments; the rest expected to have 1
      val msg: String =
        "size of aggregate's input expected to be 0 or 1 - actually is " + aggExpr.getArgList
          .size()
      throw new PlanConversionException(msg)
    }
    if (aggExpr.getArgList.size() == 1) {
      val e: JValue = emitArg(aggExpr.getArgList.get(0), f)
      val json: JValue =
        ("type", emitType(aggExpr.getType, f)) ~ ("op", opType) ~ ("e", e)
      json
    } else {
      //val e: JValue =
      val json: JValue =
        ("type", emitType(aggExpr.getType, f)) ~ ("op", opType) ~ ("e", "")
      json
    }
  }

  def emitOp(
      op: SqlOperator,
      args: ImmutableList[RexNode],
      dataType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue =
    op match {
      case binOp: SqlBinaryOperator =>
        emitBinaryOp(binOp, args, dataType, f, input)
      case func: SqlFunction => emitFunc(func, args, dataType, f, input)
      case caseOp: SqlCaseOperator =>
        emitCaseOp(caseOp, args, dataType, f, input)
      case postOp: SqlPostfixOperator =>
        emitPostfixOp(postOp, args, dataType, f, input)
      case likeOp: SqlLikeOperator =>
        throw new PlanConversionException("Unconverted like operation!")
      case _ =>
        throw new PlanConversionException("Unknown operator: " + op.getKind.sql)
    }

  def emitCast(
      arg: RexNode,
      retType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    ("expression", "cast") ~ ("type", retType) ~ ("e", emitExpression(
      arg,
      f,
      input
    ))
  }

  def emitPostfixOp(
      op: SqlPostfixOperator,
      args: ImmutableList[RexNode],
      opType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    op.getKind match {
      case SqlKind.IS_NOT_NULL =>
        ("expression", "is_not_null") ~ ("e", emitExpression(
          args.get(0),
          f,
          input
        ))
      case SqlKind.IS_NULL =>
        ("expression", "is_null") ~ ("e", emitExpression(args.get(0), f, input))
      case _ =>
        throw new PlanConversionException(
          "Unknown sql operator: " + op.getKind.sql
        )
    }
  }

  def castLeft(ltype: RelDataType, rtype: RelDataType): Boolean = {
    val l_isInt = SqlTypeUtil.isIntType(ltype)
    val r_isInt = SqlTypeUtil.isIntType(rtype)
    if (l_isInt != r_isInt) {
      if (l_isInt) {
        true
      } else {
        false
      }
    } else if (
      SqlTypeUtil.comparePrecision(ltype.getPrecision, rtype.getPrecision) < 0
    ) {
      true
    } else {
      //if they have the same precision, what should we check next?
      false
    }
  }

  def emitBinaryOp(
      op: SqlBinaryOperator,
      args: ImmutableList[RexNode],
      opType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    var left: JValue = null;
    var right: JValue = null;

    if (args.size == 2) {
      //the following casting may also be necessary in the non-binary case, but its probably safer not to do it
      //as long as we do not have examples invoking it
      val ltype = args.get(0).getType
      val rtype = args.get(1).getType

      val notnums =
        !SqlTypeUtil.isNumeric(ltype) || !SqlTypeUtil.isNumeric(rtype)

      //Binary operations containing string only operate on strings, so its safe to pass null as other

      val l_isInt = SqlTypeUtil.isIntType(ltype) || SqlTypeUtil.isBigint(ltype);
      val r_isInt = SqlTypeUtil.isIntType(rtype) || SqlTypeUtil.isBigint(rtype);
      // NOTE: should re-check all this... especially the div cases
      if (op == SqlStdOperatorTable.DIVIDE) {
        if (!l_isInt || !r_isInt) {
          // Make sure that both sides are doubles
          if (ltype.getSqlTypeName != SqlTypeName.DOUBLE) {
            left = emitCast(
              args.get(0),
              emitPrimitiveType(SqlTypeName.DOUBLE),
              f,
              input
            )
          } else {
            left = emitExpression(args.get(0), f, args.get(1), input)
          }

          if (rtype.getSqlTypeName != SqlTypeName.DOUBLE) {
            right = emitCast(
              args.get(1),
              emitPrimitiveType(SqlTypeName.DOUBLE),
              f,
              input
            )
          } else {
            right = emitExpression(args.get(1), f, args.get(0), input)
          }
        } else {
          if (castLeft(ltype, rtype)) {
            System.out.println("Cast: " + ltype + "->" + rtype)
            left = emitCast(args.get(0), emitType(rtype, f), f, input)
            right = emitExpression(args.get(1), f, args.get(0), input)
          } else {
            System.out.println("Cast: " + rtype + "->" + ltype)
            left = emitExpression(args.get(0), f, args.get(1), input)
            right = emitCast(args.get(1), emitType(ltype, f), f, input)
          }
        }
      } else if (op == SqlStdOperatorTable.DIVIDE_INTEGER) {
        if (!l_isInt || !r_isInt) {
          // If any input is not int, cast both to int(64)
          left = emitCast(
            args.get(0),
            emitPrimitiveType(SqlTypeName.BIGINT),
            f,
            input
          )
          right = emitCast(
            args.get(1),
            emitPrimitiveType(SqlTypeName.BIGINT),
            f,
            input
          )
        } else {
          //otherwise, cast to the bigger integer
          if (castLeft(ltype, rtype)) {
            System.out.println("Cast: " + ltype + "->" + rtype)
            left = emitCast(args.get(0), emitType(rtype, f), f, input)
            right = emitExpression(args.get(1), f, args.get(0), input)
          } else {
            System.out.println("Cast: " + rtype + "->" + ltype)
            left = emitExpression(args.get(0), f, args.get(1), input)
            right = emitCast(args.get(1), emitType(ltype, f), f, input)
          }
        }
        //otherwise, cast to the bigger integer
        if (castLeft(ltype, rtype)) {
          System.out.println("Cast: " + ltype + "->" + rtype)
          left = emitCast(args.get(0), emitType(rtype, f), f, input)
          right = emitExpression(args.get(1), f, args.get(0), input)
        } else {
          System.out.println("Cast: " + rtype + "->" + ltype)
          left = emitExpression(args.get(0), f, args.get(1), input)
          right = emitCast(args.get(1), emitType(ltype, f), f, input)
        }
      } else if (notnums || SqlTypeUtil.sameNamedType(ltype, rtype)) {
        left = emitExpression(args.get(0), f, args.get(1), input)
        right = emitExpression(args.get(1), f, args.get(0), input)
      } else {
//        assert(SqlTypeUtil.comparePrecision(ltype.getPrecision, rtype.getPrecision) != 0) //FIXME: !!! have to be similar, but from proteus side!
        if (castLeft(ltype, rtype)) {
          System.out.println("Cast: " + ltype + "->" + rtype)
          left = emitCast(args.get(0), emitType(rtype, f), f, input)
          right = emitExpression(args.get(1), f, args.get(0), input)
        } else {
          System.out.println("Cast: " + rtype + "->" + ltype)
          left = emitExpression(args.get(0), f, args.get(1), input)
          right = emitCast(args.get(1), emitType(ltype, f), f, input)
        }
      }
    } else {
      assert(args.size > 2);
      val subSize = (args.size + 1) / 2
      val left_args = args.subList(0, subSize)
      val right_args = args.subList(subSize, args.size)

      left = emitBinaryOp(op, left_args, opType, f, input)

      if (right_args.size == 1) {
        right = emitExpression(right_args.get(0), f, input)
      } else {
        right = emitBinaryOp(op, right_args, opType, f, input)
      }
    }

    val opName: String = op.getKind match {
      case SqlKind.GREATER_THAN          => "gt"
      case SqlKind.GREATER_THAN_OR_EQUAL => "ge"
      case SqlKind.LESS_THAN             => "lt"
      case SqlKind.LESS_THAN_OR_EQUAL    => "le"
      case SqlKind.EQUALS                => "eq"
      case SqlKind.NOT_EQUALS            => "neq"
      case SqlKind.AND                   => "and"
      case SqlKind.OR                    => "or"
      case SqlKind.TIMES                 => "multiply"
      case SqlKind.DIVIDE                => "div"
      case SqlKind.PLUS                  => "add"
      case SqlKind.MINUS                 => "sub"
      case _ =>
        if (op == SqlStdOperatorTable.DIVIDE_INTEGER) "div"
        else
          throw new PlanConversionException(
            "Unsupported binary operator: " + op.getKind.sql
          )
    }

    val json = ("expression", opName) ~ ("left", left) ~ ("right", right)
    json
  }

  def emitFunc(
      func: SqlFunction,
      args: ImmutableList[RexNode],
      retType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    val json = func.getKind match {
      case SqlKind.MOD =>
        assert(args.size == 2)
        // TODO: Return type SHOULD be the same as divisor (2nd operand)
        // SQL2003 Part2 Section 6.27, Syntax Rules 9
        var left: JValue = null;
        var right: JValue = null;

        //the following casting may also be necessary in the non-binary case, but its probably safer not to do it
        //as long as we do not have examples invoking it
        val ltype = args.get(0).getType
        val rtype = args.get(1).getType

        val notnums =
          !SqlTypeUtil.isNumeric(ltype) || !SqlTypeUtil.isNumeric(rtype)

        //Binary operations containing string only operate on strings, so its safe to pass null as other

        val l_isInt = SqlTypeUtil.isIntType(ltype)
        val r_isInt = SqlTypeUtil.isIntType(rtype)
        if (notnums || SqlTypeUtil.sameNamedType(ltype, rtype)) {
          left = emitExpression(args.get(0), f, args.get(1), input)
          right = emitExpression(args.get(1), f, args.get(0), input)
        } else {
          //        assert(SqlTypeUtil.comparePrecision(ltype.getPrecision, rtype.getPrecision) != 0) //FIXME: !!! have to be similar, but from proteus side!
          if (castLeft(ltype, rtype)) {
            System.out.println("Cast: " + ltype + "->" + rtype)
            left = emitCast(args.get(0), emitType(rtype, f), f, input)
            right = emitExpression(args.get(1), f, args.get(0), input)
          } else {
            System.out.println("Cast: " + rtype + "->" + ltype)
            left = emitExpression(args.get(0), f, args.get(1), input)
            right = emitCast(args.get(1), emitType(ltype, f), f, input)
          }
        }
        ("expression", "mod") ~ ("left", left) ~ ("right", right)
      case SqlKind.CAST =>
        assert(args.size == 1)
        emitCast(args.get(0), retType, f, input)
      case SqlKind.EXTRACT =>
        val range = args
          .get(0)
          .asInstanceOf[RexLiteral]
          .getValue
          .asInstanceOf[TimeUnitRange]
        ("expression", "extract") ~ ("unitrange", range
          .name()) ~ ("e", emitExpression(args.get(1), f, input))
      case SqlKind.OTHER_FUNCTION =>
        ("expression", func.getSqlIdentifier.toString) ~ ("argCnt", args
          .size()) /* FIXME */
      case _ =>
        throw new PlanConversionException(
          "Unsupported function: " + func.getKind.sql
        )
    }
    json
  }

  //First n-1 args: Consecutive if-then pairs
  //Last arg: 'Else' clause
  def emitCaseOp(
      op: SqlCaseOperator,
      args: ImmutableList[RexNode],
      opType: JValue,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    buildCaseExpr(
      args.asScala.dropRight(1).grouped(2).toList,
      args.get(args.size() - 1),
      f,
      input
    )
  }

  def buildCaseExpr(
      args: List[mutable.Buffer[RexNode]],
      elseNode: RexNode,
      f: List[Binding],
      input: RelNode
  ): JValue = {
    ("expression", "if") ~
      ("cond", emitExpression(args.head.head, f, input)) ~
      ("then", emitExpression(args.head(1), f, input)) ~
      ("else", if (args.size == 1) emitExpression(elseNode, f, input)
      else buildCaseExpr(args.tail, elseNode, f, input))
  }

  def emitArg(arg: RexInputRef, f: List[Binding]): JValue = {
    emitArg(
      arg,
      f,
      with_type = true
    ) // FIXME: this is confusing, the default case changes based on the overload!
  }

  def flatten(f: RelDataTypeField): List[RelDataTypeField] = {
    if (f.getType.isStruct) {
      flatten(f.getType.getFieldList.asScala.toList)
    } else {
      List(f)
    }
  }

  def flatten(fields: List[RelDataTypeField]): List[RelDataTypeField] = {
    fields.flatMap(f => flatten(f))
  }

  def findAttrInfo(
      arg: RexInputRef,
      f: List[Binding]
  ): (RelDataTypeField, PelagoTable) = {
    var fieldCount = 0
    for (b <- f) {
      val fields = b.fields //flatten(b.fields)
      fieldCount += fields.size
      if (arg.getIndex < fieldCount) {
        return (fields(arg.getIndex - (fieldCount - fields.size)), b.rel)
      }
    }
    assert(false)
    (null, null)
  }

  def emitArg(
      arg: RexInputRef,
      f: List[Binding],
      with_type: Boolean
  ): JValue = {
    val info = findAttrInfo(arg, f)
    val orig = info._1 match {
      case conv: ProjectedRelDataTypeField =>
        if (conv == null) null
        else conv.orig
      case _ =>
        null
    }
    val attr = info._1.getName
    val rel = info._2

    val finalRel: String = {
      if (orig == null) {
        rel.getPelagoRelName
      } else {
        orig
      }
    }

    val jsonAttr: JObject = {
      val json = ("attrName", attr) ~ ("relName", finalRel)

      if (with_type) {
        json ~ ("type", emitType(arg.getType, f))
      } else {
        json
      }
    }

    ("expression", "recordProjection") ~ ("e", (
      "expression",
      "argument"
    )) ~ ("attribute", jsonAttr)
  }

  def emitArg(arg: Integer, f: List[Binding]): JValue = {
    emitArg(arg, f, with_type = false)
  }

  def emitPrimitiveType(k: SqlTypeName): JValue =
    k match {
      case SqlTypeName.INTEGER => ("type", "int")
      case SqlTypeName.BIGINT  => ("type", "int64")
      case SqlTypeName.VARCHAR =>
        ("type", if (dictEncoded) "dstring" else "string")
      case SqlTypeName.CHAR =>
        ("type", if (dictEncoded) "dstring" else "string")
      case SqlTypeName.BOOLEAN   => ("type", "bool")
      case SqlTypeName.DATE      => ("type", "date")
      case SqlTypeName.TIMESTAMP => ("type", "datetime")
      case SqlTypeName.DOUBLE =>
        ("type", "float") // proteu's float is a c++ double
      case SqlTypeName.FLOAT =>
        ("type", "float") // proteu's float is a c++ double
      case SqlTypeName.DECIMAL => ("type", "float")
      case _                   => throw new PlanConversionException("Unknown type: " + k)
    }

  def emitType(arg: RelDataType, binding: List[Binding]): JValue =
    arg.getSqlTypeName match {
      case SqlTypeName.ARRAY =>
        ("type", "list") ~ ("inner", emitType(arg.getComponentType, binding))
      case SqlTypeName.ROW =>
        val rel = binding.head.rel
        val b = List(
          Binding(
            rel,
            arg.asInstanceOf[RelRecordType].getFieldList.asScala.toList
          )
        )
        ("type", "record") ~ ("attributes",
        arg.getFieldList.asScala.map { f =>
          emitArg(f.getIndex, b, with_type = true)
        })
      case SqlTypeName.MULTISET =>
        ("type", "bag") ~ ("inner", emitType(arg.getComponentType, binding))
      case _ => emitPrimitiveType(arg.getSqlTypeName)
    }

  def emitArg(arg: Integer, f: List[Binding], with_type: Boolean): JValue = {
    var rel: PelagoTable = null
    var attr: String = ""
    var t: JValue = null
    var fieldCount = 0
    var fieldCountCurr = 0
    breakable {
      for (b <- f) {
        fieldCount += b.fields.size
        if (arg < fieldCount) {
          rel = b.rel
          attr = b.fields(arg - fieldCountCurr).getName
          if (with_type)
            t = emitType(b.fields(arg - fieldCountCurr).getType, List(b))
          break
        }
        fieldCountCurr += b.fields.size
      }
    }

    val json: JObject = ("relName", rel.getPelagoRelName) ~ ("attrName", attr)
    if (with_type) {
      json ~ ("type", t)
    } else {
      json
    }
  }

  def emitSchema(relName: PelagoTable, t: RelDataType): JValue = {
    emitSchema(
      relName,
      t,
      with_attrNo = false,
      is_packed = false,
      with_type = false
    )
  }

  def emitSchema(
      relName: PelagoTable,
      t: RelDataType,
      with_attrNo: Boolean,
      is_packed: Boolean
  ): JValue = {
    emitSchema(relName, t, with_attrNo, is_packed, with_type = false)
  }

  def emitSchema(
      relName: PelagoTable,
      t: RelDataType,
      with_attrNo: Boolean,
      is_packed: Boolean,
      with_type: Boolean
  ): JValue =
    t match {
      case recType: RelRecordType =>
        emitRowType(relName, recType, with_attrNo, is_packed, with_type)
      case _ =>
        throw new PlanConversionException(
          "Unknown schema type (non-record one)"
        )
    }

  def emitRowType(relName: PelagoTable, t: RelRecordType): JValue = {
    emitRowType(
      relName,
      t,
      with_attrNo = false,
      is_packed = false,
      with_type = false
    )
  }

  def emitRowType(
      relName: PelagoTable,
      t: RelRecordType,
      with_attrNo: Boolean,
      is_packed: Boolean,
      with_type: Boolean
  ): JValue = {
    val bindings = List(Binding(relName, getFields(t)))
    val fields = t.getFieldList.asScala.zipWithIndex.map { f =>
      {
        var t =
          ("relName", relName.getPelagoRelName) ~ ("attrName", f._1.getName)
        if (with_type) {
          t = t ~ ("type", emitType(
            f._1.getType,
            bindings
          ))
        }
        if (with_attrNo) t = t ~ ("attrNo", f._2 + 1)
        if (is_packed) t = t ~ ("isBlock", true)
        t
      }
    }
    fields
  }
}
