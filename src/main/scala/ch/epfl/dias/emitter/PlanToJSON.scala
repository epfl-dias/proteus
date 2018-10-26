package ch.epfl.dias.emitter

import java.io.{PrintWriter, StringWriter}

import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan
import ch.epfl.dias.emitter.PlanToJSON.emitPrimitiveType
import com.google.common.collect.ImmutableList
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.avatica.util.{DateTimeUtils, TimeUnit, TimeUnitRange}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.{RelDataType, RelDataTypeFactory, RelDataTypeField, RelRecordType}
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rex._
import org.apache.calcite.sql.fun.{SqlCaseOperator, SqlCastFunction, SqlLikeOperator, SqlStdOperatorTable}
import org.apache.calcite.sql.{pretty => _, _}
import org.apache.calcite.interpreter.Bindables.BindableTableScan
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.RelOptUtil.InputFinder
import org.apache.calcite.sql.`type`.{ArraySqlType, SqlTypeName, SqlTypeUtil}
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable.Buffer
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

case class Binding(rel: String, fields: List[RelDataTypeField])

object PlanToJSON {
  implicit val formats = DefaultFormats

  //-> TODO What about degree of materialization for joins?
  //---> Join behavior can be handled implicitly by executor:
  //-----> Early materialization: Create new OID
  //-----> Late materialization: Preserve OID + rowTypes of join's children instead of the new one
  //-> TODO Do we need projection pushdown at this level?

  def jsonToString(in: JValue) : String = {
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

  def getFields(t: RelDataType) : List[RelDataTypeField] = t match {
    case recType : RelRecordType => recType.getFieldList.asScala.toList
    case _ => {
      val msg : String = "getFields() assumes RecordType as default input"
      throw new PlanConversionException(msg)
      //List[RelDataTypeField]()
    }
  }

  def emitExpression(e: RexNode, f: List[Binding]) : JValue = {
    emitExpression(e, f, false)
  }

  def emitExpression(e: RexNode, f: List[Binding], other: RexNode) : JValue = {
    emitExpression(e, f, other, false)
  }

  def emitExpression(e: RexNode, f: List[Binding], arg_with_type: Boolean) : JValue = {
    emitExpression(e, f, null, arg_with_type)
  }

  def emitExpression(e: RexNode, f: List[Binding], other: RexNode, arg_with_type: Boolean) : JValue = {
    val exprType : JValue = emitType(e.getType, f)
    val json : JValue = e match {
      case call: RexCall => {
        emitOp(call.op, call.operands, exprType, f)
      }
      case inputRef: RexInputRef => {
        val arg = emitArg(inputRef,f, arg_with_type)
        arg
      }
      case lit: RexLiteral => {
        if (lit.isNull) {
          ("expression", exprType \ "type") ~ ("isNull", true)
        } else {
          val v: JValue = lit.getType.getSqlTypeName match {
            case SqlTypeName.INTEGER => new Integer(lit.toString).asInstanceOf[Int]
            case SqlTypeName.BIGINT => new java.lang.Long(lit.toString).asInstanceOf[Long]
            case SqlTypeName.BOOLEAN => new java.lang.Boolean(lit.toString).asInstanceOf[Boolean]
            case SqlTypeName.FLOAT => new java.lang.Double(lit.toString).asInstanceOf[Double]
            case SqlTypeName.DOUBLE => new java.lang.Double(lit.toString).asInstanceOf[Double]
            case SqlTypeName.DECIMAL => new java.lang.Double(lit.toString).asInstanceOf[Double]
            case SqlTypeName.DATE => new java.lang.Long(DateTimeUtils.timestampStringToUnixDate(lit.toString)).asInstanceOf[Long]
            case SqlTypeName.VARCHAR => lit.getValueAs(classOf[String]) //.toString.substring(1, lit.to)
            case SqlTypeName.CHAR => lit.getValueAs(classOf[String])
            case _ => {
              val msg: String = "Unknown constant type"
              throw new PlanConversionException(msg)
              //List[RelDataTypeField]()
            }
          }
          if (other != null && lit.getType.getSqlTypeName == SqlTypeName.VARCHAR || lit.getType.getSqlTypeName == SqlTypeName.CHAR) {
            // Only comparisons of input fields with string constants are supported
            // otherwise, which dictionary should we use?

            val info = findAttrInfo(other.asInstanceOf[RexInputRef], f)
            val path = info._2 + "." + info._1 + ".dict"

            ("expression", exprType \ "type") ~ ("v", v) ~ ("dict", ("path", path))
          } else {
            ("expression", exprType \ "type") ~ ("v", v)
          }
        }
      }
      case _ => {
        val msg : String = "Unsupported expression "+e.toString
        throw new PlanConversionException(msg)
      }
    }
    json
  }

  def aggKind(agg: SqlAggFunction): String = agg.getKind match {
    case SqlKind.AVG     => "avg"
    case SqlKind.COUNT   => "sum"
    case SqlKind.MAX     => "max"
    case SqlKind.MIN     => "min"
    case SqlKind.SUM     => "sum"
    //'Sum0 is an aggregator which returns the sum of the values which go into it like Sum.'
    //'It differs in that when no non null values are applied zero is returned instead of null.'
    case SqlKind.SUM0    => "sum"
    case SqlKind.COLLECT => "bagunion"
    case _ => {
      val msg : String = "unknown aggr. function " + agg.getKind.toString
      throw new PlanConversionException(msg)
    }
  }


  //TODO Assumming unary ops
  def emitAggExpression(aggExpr: AggregateCall, f: List[Binding]) : JValue = {
    val opType : String = aggKind(aggExpr.getAggregation)

    if(aggExpr.getArgList.size() > 1)  {
      //count() has 0 arguments; the rest expected to have 1
      val msg : String = "size of aggregate's input expected to be 0 or 1 - actually is "+aggExpr.getArgList.size()
      throw new PlanConversionException(msg)
    }
    if(aggExpr.getArgList.size() == 1)  {
      val e: JValue = emitArg(aggExpr.getArgList.get(0),f)
      val json : JValue = ("type", emitType(aggExpr.getType, f)) ~ ("op",opType) ~ ("e",e)
      json
    } else  {
      //val e: JValue =
      val json : JValue = ("type", emitType(aggExpr.getType, f)) ~ ("op",opType) ~ ("e","")
      json
    }
  }

  def emitOp(op: SqlOperator, args: ImmutableList[RexNode], dataType: JValue, f: List[Binding] ) : JValue = op match   {
    case binOp: SqlBinaryOperator => emitBinaryOp(binOp, args, dataType, f)
    case func: SqlFunction => emitFunc(func, args, dataType, f)
    case caseOp: SqlCaseOperator => emitCaseOp(caseOp, args, dataType, f)
    case postOp: SqlPostfixOperator => emitPostfixOp(postOp, args, dataType, f)
    case likeOp: SqlLikeOperator => throw new PlanConversionException("Unconverted like operation!")
    case _ => throw new PlanConversionException("Unknown operator: "+op.getKind.sql)
  }

  def emitCast(arg: RexNode, retType: JValue, f: List[Binding]): JValue = {
    ("expression", "cast") ~ ("type", retType) ~ ("e", emitExpression(arg, f))
  }

  def emitPostfixOp(op: SqlPostfixOperator, args: ImmutableList[RexNode], opType: JValue, f: List[Binding]) : JValue = {
    op.getKind match {
      case SqlKind.IS_NOT_NULL => ("expression", "is_not_null") ~ ("e", emitExpression(args.get(0), f))
      case SqlKind.IS_NULL     => ("expression", "is_null"    ) ~ ("e", emitExpression(args.get(0), f))
      case _ => throw new PlanConversionException("Unknown sql operator: "+op.getKind.sql)
    }
  }

  def castLeft(ltype: RelDataType, rtype: RelDataType): Boolean = {
    val l_isInt = SqlTypeUtil.isIntType(ltype)
    val r_isInt = SqlTypeUtil.isIntType(rtype)
    if (l_isInt != r_isInt){
      if (l_isInt){
        true
      } else {
        false
      }
    } else if (SqlTypeUtil.comparePrecision(ltype.getPrecision, rtype.getPrecision) < 0){
      true
    } else {
      //if they have the same precision, what should we check next?
      false
    }
  }

  def emitBinaryOp(op: SqlBinaryOperator, args: ImmutableList[RexNode], opType: JValue, f: List[Binding]) : JValue =  {
    var left : JValue = null;
    var right: JValue = null;

    if (args.size == 2){
      //the following casting may also be necessary in the non-binary case, but its probably safer not to do it
      //as long as we do not have examples invoking it
      val ltype = args.get(0).getType
      val rtype = args.get(1).getType

      val notnums = !SqlTypeUtil.isNumeric(ltype) || !SqlTypeUtil.isNumeric(rtype)

      //Binary operations containing string only operate on strings, so its safe to pass null as other

      val l_isInt   = SqlTypeUtil.isIntType(ltype)
      val r_isInt   = SqlTypeUtil.isIntType(rtype)
      // NOTE: should re-check all this... especially the div cases
      if (op == SqlStdOperatorTable.DIVIDE) {
        // Make sure that both sides are doubles
        if (ltype.getSqlTypeName != SqlTypeName.DOUBLE) {
          left = emitCast(args.get(0), emitPrimitiveType(SqlTypeName.DOUBLE), f)
        } else {
          left  = emitExpression(args.get(0), f, args.get(1))
        }

        if (rtype.getSqlTypeName != SqlTypeName.DOUBLE) {
          right = emitCast(args.get(1), emitPrimitiveType(SqlTypeName.DOUBLE), f)
        } else {
          right = emitExpression(args.get(1), f, args.get(0))
        }
      } else if (op == SqlStdOperatorTable.DIVIDE_INTEGER) {
        if (!l_isInt || !r_isInt){
          // If any input is not int, cast both to int(64)
          left  = emitCast(args.get(0), emitPrimitiveType(SqlTypeName.BIGINT), f)
          right = emitCast(args.get(1), emitPrimitiveType(SqlTypeName.BIGINT), f)
        } else {
          //otherwise, cast to the bigger integer
          if (castLeft(ltype, rtype)){
            System.out.println("Cast: " + ltype + "->" + rtype)
            left  = emitCast      (args.get(0), emitType(rtype, f), f)
            right = emitExpression(args.get(1), f, args.get(0))
          } else {
            System.out.println("Cast: " + rtype + "->" + ltype)
            left  = emitExpression(args.get(0), f, args.get(1))
            right = emitCast      (args.get(1), emitType(ltype, f), f)
          }
        }
      } else if (notnums || SqlTypeUtil.sameNamedType(ltype, rtype)) {
        left  = emitExpression(args.get(0), f, args.get(1))
        right = emitExpression(args.get(1), f, args.get(0))
      } else {
//        assert(SqlTypeUtil.comparePrecision(ltype.getPrecision, rtype.getPrecision) != 0) //FIXME: !!! have to be similar, but from proteus side!
        if (castLeft(ltype, rtype)){
          System.out.println("Cast: " + ltype + "->" + rtype)
          left  = emitCast      (args.get(0), emitType(rtype, f), f)
          right = emitExpression(args.get(1), f, args.get(0))
        } else {
          System.out.println("Cast: " + rtype + "->" + ltype)
          left  = emitExpression(args.get(0), f, args.get(1))
          right = emitCast      (args.get(1), emitType(ltype, f), f)
        }
      }
    } else {
      assert(args.size > 2);
      val subSize = (args.size + 1) / 2
      val left_args = args.subList(0, subSize)
      val right_args = args.subList(subSize, args.size)

      left = emitBinaryOp(op, left_args, opType, f)

      if (right_args.size == 1){
        right = emitExpression(right_args.get(0), f)
      } else {
        right = emitBinaryOp(op, right_args, opType, f)
      }
    }

    val opName : String = op.getKind match {
      case SqlKind.GREATER_THAN => "gt"
      case SqlKind.GREATER_THAN_OR_EQUAL => "ge"
      case SqlKind.LESS_THAN => "lt"
      case SqlKind.LESS_THAN_OR_EQUAL => "le"
      case SqlKind.EQUALS => "eq"
      case SqlKind.NOT_EQUALS => "neq"
      case SqlKind.AND => "and"
      case SqlKind.OR => "or"
      case SqlKind.TIMES => "multiply"
      case SqlKind.DIVIDE => "div"
      case SqlKind.PLUS => "add"
      case SqlKind.MINUS => "sub"
      case _ => {
        if (op == SqlStdOperatorTable.DIVIDE_INTEGER) "div"
        else throw new PlanConversionException("Unsupported binary operator: "+op.getKind.sql)
      }
    }

    val json = ("expression",opName) ~ ("left", left) ~ ("right", right)
    json
  }
  

  def emitFunc(func: SqlFunction, args: ImmutableList[RexNode], retType: JValue, f: List[Binding]) : JValue =  {
    val json = func.getKind match  {
      case SqlKind.CAST => {
        assert(args.size == 1)
        emitCast(args.get(0), retType, f)
      }
      case SqlKind.EXTRACT => {
        val range = args.get(0).asInstanceOf[RexLiteral].getValue.asInstanceOf[TimeUnitRange]
        ("expression", "extract") ~ ("unitrange", range.name()) ~ ("e", emitExpression(args.get(1), f))
      }
      case _ => throw new PlanConversionException("Unsupported function: "+func.getKind.sql)
    }
    json
  }

  //First n-1 args: Consecutive if-then pairs
  //Last arg: 'Else' clause
  def emitCaseOp(op: SqlCaseOperator, args: ImmutableList[RexNode], opType: JValue, f: List[Binding]) : JValue =  {
    buildCaseExpr(args.asScala.dropRight(1).grouped(2).toList, args.get(args.size() - 1), f)
  }

  def buildCaseExpr(args: List[Buffer[RexNode]], elseNode: RexNode, f: List[Binding]): JValue = {
    ("expression", "if") ~
      ("cond", emitExpression(args(0)(0), f)) ~
      ("then", emitExpression(args(0)(1), f)) ~
      ("else", if (args.size == 1) emitExpression(elseNode, f) else buildCaseExpr(args.tail, elseNode, f))
  }

  def emitArg(arg: RexInputRef, f: List[Binding]) : JValue = {
    emitArg(arg, f, true) // FIXME: this is confusing, the default case changes based on the overload!
  }

  def findAttrInfo(arg: RexInputRef, f: List[Binding]) = {
    var rel : String = ""
    var attr : String = ""
    var fieldCount = 0
    var fieldCountCurr = 0
    breakable { for(b <- f) {
      fieldCount += b.fields.size
      if(arg.getIndex < fieldCount)  {
      rel = b.rel
      attr = b.fields(arg.getIndex - fieldCountCurr).getName
      break
    }
      fieldCountCurr += b.fields.size
    } }
    (attr, rel)
  }

  def emitArg(arg: RexInputRef, f: List[Binding], with_type: Boolean) : JValue = {
    val info = findAttrInfo(arg, f)
    val attr = info._1
    val rel  = info._2

    val jsonAttr: JObject = {
      val json = ("attrName", attr) ~ ("relName", rel)

      if (with_type){
        json ~ ("type", emitType(arg.getType, f))
      } else {
        json
      }
    }

    val jsonArg: JObject =
      ("expression" , "argument"                            ) ~
      ("attributes" , List(jsonAttr)                        ) ~
      ("type"       , ("relName", rel) ~ ("type", "record") ) ~
      ("argNo"      , -1                                    )


    ("expression", "recordProjection") ~ ("e", jsonArg) ~ ("attribute", jsonAttr)
  }

  def emitArg(arg: Integer, f: List[Binding]) : JValue = {
    emitArg(arg, f, false)
  }

  def emitPrimitiveType(k: SqlTypeName): JValue = k match {
    case SqlTypeName.INTEGER  => ("type", "int"     )
    case SqlTypeName.BIGINT   => ("type", "int64"   )
    case SqlTypeName.VARCHAR  => ("type", "dstring" )
    case SqlTypeName.CHAR     => ("type", "dstring" )
    case SqlTypeName.BOOLEAN  => ("type", "bool"    )
    case SqlTypeName.DATE     => ("type", "date"    )
    case SqlTypeName.DOUBLE   => ("type", "float"   ) // proteu's float is a c++ double
    case SqlTypeName.FLOAT    => ("type", "float"   ) // proteu's float is a c++ double
    case SqlTypeName.DECIMAL  => ("type", "float"   )
    case _ => throw new PlanConversionException("Unknown type: " + k)
  }


  def emitType(arg: RelDataType, binding: List[Binding]): JValue = arg.getSqlTypeName match {
    case SqlTypeName.ARRAY   => {
      ("type", "list") ~ ("inner", emitType(arg.getComponentType, binding))
    }
    case SqlTypeName.ROW     => {
      ("type", "record") ~ ("attributes",
        arg.getFieldList.asScala.map{ f =>
          emitArg(f.getIndex, binding, true)
        }
      )
    }
    case SqlTypeName.MULTISET   => {
      ("type", "bag") ~ ("inner", emitType(arg.getComponentType, binding))
    }
    case _ => emitPrimitiveType(arg.getSqlTypeName)
  }

  def emitArg(arg: Integer, f: List[Binding], with_type: Boolean) : JValue = {
    var rel : String = ""
    var attr : String = ""
    var t : JValue = null
    var fieldCount = 0
    var fieldCountCurr = 0
    breakable { for(b <- f) {
      fieldCount += b.fields.size
      if(arg < fieldCount)  {
        rel = b.rel
        attr = b.fields(arg - fieldCountCurr).getName
        if (with_type) t = emitType(b.fields(arg - fieldCountCurr).getType, List(b))
        break
      }
      fieldCountCurr += b.fields.size
    } }

    val json : JObject = ("relName",rel) ~ ("attrName",attr)
    if (with_type){
      json ~ ("type", t)
    } else {
      json
    }
  }

  def emitSchema(relName: String, t: RelDataType): JValue = {
    emitSchema(relName, t, false, false, false)
  }

  def emitSchema(relName: String, t: RelDataType, with_attrNo: Boolean, is_packed: Boolean): JValue = {
    emitSchema(relName, t, with_attrNo, is_packed, false)
  }

  def emitSchema(relName: String, t: RelDataType, with_attrNo: Boolean, is_packed: Boolean, with_type: Boolean): JValue = t match {
    case recType : RelRecordType => emitRowType(relName, recType, with_attrNo, is_packed, with_type)
    case _ => throw new PlanConversionException("Unknown schema type (non-record one)")
  }

  def emitRowType(relName: String, t: RelRecordType): JValue = {
    emitRowType(relName, t, false, false, false)
  }

  def emitRowType(relName: String, t: RelRecordType, with_attrNo: Boolean, is_packed: Boolean, with_type: Boolean): JValue = {
    val bindings = List(Binding(relName, getFields(t)))
    val fields = t.getFieldList.asScala.zipWithIndex.map {
      f => {
        var t = ("relName", relName) ~ ("attrName", f._1.getName)
        if (with_type) {
//          var ty = emitType(f._1.getType, bindings)
//          if (f._1.getType.getSqlTypeName == SqlTypeName.VARCHAR || f._1.getType.getSqlTypeName == SqlTypeName.CHAR){
//            ty = ty.asInstanceOf[JObject] ~ ("dict", ("path", relName + "." + f._1.getName + ".dict"))
//          }
          t = t ~ ("type", emitType(f._1.getType, bindings)) //  ("dict", ("path", path)))
        }
        if (with_attrNo) t = t ~ ("attrNo", f._2 + 1)
        if (is_packed  ) t = t ~ ("isBlock", true)
        t
      }
    }
    fields
  }

  def emit(n: RelNode) : JValue = {
    emit_(n)._2
  }

  def emit_(n: RelNode) : (Binding, JValue) = n match {
    //Note: Project can appear in multiple parts of a plan!
    //For example: if we use 'GROUP BY id+9, a Project operator will put the new bindings together
//    case p: EnumerableProject => {
//      val op = ("operator" , "projection")
//      val alias = "projection"+p.getId
//      val rowType = emitSchema(alias, p.getRowType)
//      val child = emit_(p.getInput)
//      val childBinding: Binding = child._1
//      val childOp = child._2
//      //TODO Could also use p.getNamedProjects
//      val exprs = p.getProjects
//      val exprsJS: JValue = exprs.asScala.map {
//        e => emitExpression(e,List(childBinding))
//      }
//
//      val json = op ~ ("tupleType", rowType) ~ ("e", exprsJS) ~ ("input" , childOp)
//      val binding: Binding = Binding(alias,getFields(p.getRowType))
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
//    case a: EnumerableAggregate => {
//      val op = ("operator" , "agg")
//      val child = emit_(a.getInput)
//      val childBinding: Binding = child._1
//      val childOp = child._2
//
//      val groups: List[Integer] = a.getGroupSet.toList.asScala.toList
//      val groupsJS: JValue = groups.map {
//        g => emitArg(g,List(childBinding))
//      }
//
//      val aggs: List[AggregateCall] = a.getAggCallList.asScala.toList
//      val aggsJS = aggs.map {
//        agg => emitAggExpression(agg,List(childBinding))
//      }
//      val alias = "agg"+a.getId
//      val rowType = emitSchema(alias, a.getRowType)
//
//      val json = op ~ ("tupleType", rowType) ~ ("groups", groupsJS) ~ ("aggs", aggsJS) ~ ("input" , childOp)
//      val binding: Binding = Binding(alias,getFields(a.getRowType))
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
//    case j: EnumerableJoin => {
//      val op = ("operator" , "join")
//      val l = emit_(j.getLeft)
//      val leftBinding: Binding = l._1
//      val leftChildOp = l._2
//      val r = emit_(j.getRight)
//      val rightBinding: Binding = r._1
//      val rightChildOp = r._2
//      val cond = emitExpression(j.getCondition,List(leftBinding,rightBinding))
//      val alias = "join"+j.getId
//      val rowType = emitSchema(alias, j.getRowType)
//
//      val json = op ~ ("tupleType", rowType) ~ ("cond", cond) ~ ("left" , leftChildOp) ~ ("right" , rightChildOp)
//      val binding: Binding = Binding(alias,leftBinding.fields ++ rightBinding.fields)
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
//    case f: EnumerableFilter => {
//      val op = ("operator" , "select")
//      val child = emit_(f.getInput)
//      val childBinding: Binding = child._1
//      val childOp = child._2
//      val rowType = emitSchema(childBinding.rel, f.getRowType)
//      val cond = emitExpression(f.getCondition,List(childBinding))
//
//      val json = op ~ ("tupleType", rowType) ~ ("cond", cond) ~ ("input", childOp)
//      val ret: (Binding, JValue) = (childBinding,json)
//      ret
//    }
//    case s : EnumerableTableScan => {
//      val op = ("operator" , "scan")
//      //TODO Cross-check: 0: schemaName, 1: tableName (?)
//      val srcName = s.getTable.getQualifiedName.get(1)
//      val rowType = emitSchema(srcName, s.getRowType)
//
//      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName)
//      val binding: Binding = Binding(srcName,getFields(s.getRowType))
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
//    case s : PelagoTableScan => {
//      val op = ("operator" , "scan")
//      //TODO Cross-check: 0: schemaName, 1: tableName (?)
//      val srcName  = s.getPelagoRelName //s.getTable.getQualifiedName.get(1)
//      val rowType  = emitSchema(srcName, s.getRowType)
//      val plugin   = Extraction.decompose(s.getPluginInfo.asScala)
//      val linehint = s.getLineHint.longValue
//
//      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName) ~ ("plugin", plugin) ~ ("linehint", linehint)
//      val binding: Binding = Binding(srcName,getFields(s.getRowType))
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
//    case s : BindableTableScan => {
//      val op = ("operator" , "scan")
//      //TODO Cross-check: 0: schemaName, 1: tableName (?)
//      val srcName = s.getTable.getQualifiedName.get(1)
//      val rowType = emitSchema(srcName, s.getRowType)
//
//      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName)
//      val binding: Binding = Binding(srcName,getFields(s.getRowType))
//      val ret: (Binding, JValue) = (binding,json)
//      ret
//    }
    case i : EnumerableInterpreter => {
      emit_(i.getInput)
    }
    case sort: EnumerableSort => {
      val op = ("operator" , "sort")
      val child = emit_(sort.getInput)
      val childBinding: Binding = child._1
      val childOp = child._2
      val rowType = emitSchema("sort"+sort.getId, sort.getRowType)

      val args : JValue =
      sort.getCollation.getFieldCollations.asScala.map {
        col => {
          val arg = emitArg(col.getFieldIndex,List(childBinding))
          val regas = ("rel", "sort"+sort.getId) ~ ("attr", arg.\("attr"))
          ("expression", arg) ~ ("direction", col.getDirection.shortString) ~ ("register_as", regas)
        }
      }

      val keyIndexes =
      sort.getCollation.getFieldCollations.asScala.map {
       col => col.getFieldIndex
      }

      val args2: JValue =
      sort.getRowType.getFieldList.asScala.filter(p => !keyIndexes.contains(p.getIndex)).map{
        col => {
          val arg = emitArg(col.getIndex,List(childBinding))
          val regas = ("rel", "sort"+sort.getId) ~ ("attr", arg.\("attr"))
          ("expression", arg) ~ ("direction", "NONE") ~ ("register_as", regas)
        }
      }

      val json : JValue = op ~ ("tupleType", rowType) ~ ("args", args ++ args2) ~ ("input", childOp)
      val binding: Binding = Binding("sort"+sort.getId,getFields(sort.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case c: EnumerableCorrelate => {
      val op = ("operator" , "unnest")
      val l = emit_(c.getLeft)
      val leftBinding: Binding = l._1
      val leftChildOp = l._2
      val proj = c.getRight.asInstanceOf[EnumerableUncollect].getInput.asInstanceOf[EnumerableProject]

      val unnest_exprs = proj.getNamedProjects.asScala.map {
        p => {
          val expr = p.left.asInstanceOf[RexFieldAccess]
          assert(expr.getReferenceExpr().asInstanceOf[RexCorrelVariable].id == c.getCorrelationId)
          val f = emitArg(p.left.asInstanceOf[RexFieldAccess].getField.getIndex, List(leftBinding))

          ("e", f) ~ ("name", "__unnest".concat(c.getId.toString).concat("_").concat(f.\("attr").extract[String]))
        }
      }
      val alias = "unnest"+c.getId
      val rowType = emitSchema("unnest"+c.getId, c.getRowType)

      val proj_exprs = leftChildOp\"tupleType"
      val nested_exprs = unnest_exprs.map{
        p => {
          ("attr", p\"name") ~ ("rel", (p\"e"\"rel").extract[String].concat(".").concat((p\"e"\"attr").extract[String]))
        }
      }

      val unnest_json = op ~ ("tupleType", proj_exprs ++ nested_exprs) ~ ("input" , leftChildOp) ~ ("path" , unnest_exprs)

      val unnest_attr = (nested_exprs.head\"rel").extract[String]
      val tmpBinding: Binding = Binding(unnest_attr, getFields(c.getRowType))

      val unnest_e: JValue = getFields(c.getRowType).slice(c.getLeft.getRowType.getFieldCount, getFields(c.getRowType).size).map{
        p => {
          emitArg(p.getIndex, List(tmpBinding), true) // ~ ("rel", (p\"e"\"rel").extract[String].concat(".").concat((p\"e"\"attr").extract[String]))
        }
      }

      val json = ("operator", "projection") ~ ("tupleType", rowType) ~ ("e", proj_exprs ++ unnest_e) ~ ("input" , unnest_json)

      val binding: Binding = Binding("unnest"+c.getId,getFields(c.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case _  => {
      throw new PlanConversionException("Unknown algebraic operator: "+n.getRelTypeName)
    }
  }
}
