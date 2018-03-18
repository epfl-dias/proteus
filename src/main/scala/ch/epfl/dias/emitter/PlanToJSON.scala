package ch.epfl.dias.emitter

import java.io.{PrintWriter, StringWriter}

import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan
import com.google.common.collect.ImmutableList
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.{RelDataType, RelDataTypeField, RelRecordType}
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rex.{RexCall, RexInputRef, RexLiteral, RexNode}
import org.apache.calcite.sql.fun.SqlCaseOperator
import org.apache.calcite.sql.{SqlBinaryOperator, SqlFunction, SqlKind, SqlOperator}
import org.apache.calcite.interpreter.Bindables.BindableTableScan
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

object PlanToJSON {
  implicit val formats = DefaultFormats

  case class Binding(rel: String, fields: List[RelDataTypeField])

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
    val exprType : String = e.getType.toString
    val json : JValue = e match {
      case call: RexCall => {
        emitOp(call.op, call.operands, exprType, f)
      }
      case inputRef: RexInputRef => {
        val arg = emitArg(inputRef,f)
        arg
      }
      case lit: RexLiteral => ("expression",exprType) ~ ("v",toJavaString(lit))
      case _ => {
        val msg : String = "Unsupported expression "+e.toString
        throw new PlanConversionException(msg)
      }
    }
    json
  }

  //TODO Assumming unary ops
  def emitAggExpression(aggExpr: AggregateCall, f: List[Binding]) : JValue = {
    val opType : String = aggExpr.getAggregation.getKind match {
      case SqlKind.AVG    => "avg"
      case SqlKind.COUNT  => "cnt"
      case SqlKind.MAX    => "max"
      case SqlKind.MIN    => "min"
      case SqlKind.SUM    => "sum"
      //'Sum0 is an aggregator which returns the sum of the values which go into it like Sum.'
      //'It differs in that when no non null values are applied zero is returned instead of null.'
      case SqlKind.SUM0    => "sum0"
      case _ => {
        val msg : String = "unknown aggr. function "+aggExpr.getAggregation.getKind.toString
        throw new PlanConversionException(msg)
      }
    }

    if(aggExpr.getArgList.size() > 1)  {
      //count() has 0 arguments; the rest expected to have 1
      val msg : String = "size of aggregate's input expected to be 0 or 1 - actually is "+aggExpr.getArgList.size()
      throw new PlanConversionException(msg)
    }
    if(aggExpr.getArgList.size() == 1)  {
      val e: JValue = emitArg(aggExpr.getArgList.get(0),f)
      val json : JValue = ("type", aggExpr.getType.toString) ~ ("op",opType) ~ ("e",e)
      json
    } else  {
      //val e: JValue =
      val json : JValue = ("type", aggExpr.getType.toString) ~ ("op",opType) ~ ("e","")
      json
    }
  }

  def emitOp(op: SqlOperator, args: ImmutableList[RexNode], dataType: String, f: List[Binding] ) : JValue = op match   {
    case binOp: SqlBinaryOperator => emitBinaryOp(binOp, args, dataType, f)
    case func: SqlFunction => emitFunc(func, args, dataType, f)
    case caseOp: SqlCaseOperator => emitCaseOp(caseOp, args, dataType, f)
    case _ => throw new PlanConversionException("Unknown operator: "+op.getKind.sql)
  }

  def emitBinaryOp(op: SqlBinaryOperator, args: ImmutableList[RexNode], opType: String, f: List[Binding]) : JValue =  {
    var left : JValue = null;
    var right: JValue = null;

    if (args.size == 2){
      left  = emitExpression(args.get(0), f)
      right = emitExpression(args.get(1), f)
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
      case SqlKind.TIMES => "mult"
      case SqlKind.DIVIDE => "div"
      case SqlKind.PLUS => "add"
      case SqlKind.MINUS => "sub"
      case _ => throw new PlanConversionException("Unsupported binary operator: "+op.getKind.sql)
    }

    val json = ("type", opType) ~ ("expression",opName) ~ ("left", left) ~ ("right", right)
    json
  }

  def emitFunc(func: SqlFunction, args: ImmutableList[RexNode], retType: String, f: List[Binding]) : JValue =  {
    val json = func.getKind match  {
      case SqlKind.CAST => {
        val funcName = "cast"
        val arg : JValue = List(emitExpression(args.get(0), f))
        ("type",retType) ~ ("expression",funcName) ~ ("args", arg)
      }
      case _ => throw new PlanConversionException("Unsupported function: "+func.getKind.sql)
    }
    json
  }

  //First n-1 args: Consecutive if-then pairs
  //Last arg: 'Else' clause
  def emitCaseOp(op: SqlCaseOperator, args: ImmutableList[RexNode], opType: String, f: List[Binding]) : JValue =  {
    val cases: JValue = args.asScala.dropRight(1).grouped(2).toList.map  {
      case ArrayBuffer(first: RexNode,second: RexNode) => ("if",emitExpression(first,f)) ~ ("then",emitExpression(second,f))
    }
    val elseNode : RexNode = args.get(args.size() - 1)
    val json : JValue = ("expression", "case") ~ ("cases", cases) ~ ("else", emitExpression(elseNode,f))
    json
  }

  def emitArg(arg: RexInputRef, f: List[Binding]) : JValue = {
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

    val json : JObject = ("expression" -> "argument") ~ ("type",arg.getType.toString) ~ ("rel",rel) ~ ("attr",attr)
    json
  }

  def emitArg(arg: Integer, f: List[Binding]) : JValue = {
    var rel : String = ""
    var attr : String = ""
    var fieldCount = 0
    var fieldCountCurr = 0
    breakable { for(b <- f) {
      fieldCount += b.fields.size
      if(arg < fieldCount)  {
        rel = b.rel
        attr = b.fields(arg - fieldCountCurr).getName
        break
      }
      fieldCountCurr += b.fields.size
    } }

    val json : JObject = ("rel",rel) ~ ("attr",attr)
    json
  }

  def emitSchema(relName: String, t: RelDataType): JValue = t match {
    case recType : RelRecordType => emitRowType(relName, recType)
    case _ => throw new PlanConversionException("Unknown schema type (non-record one)")
  }

  def emitRowType(relName: String, t: RelRecordType): JValue = {
    val fields = t.getFieldList.asScala.map {
      f => {
        ("rel", relName) ~ ("attr", f.getName) ~ ("type", f.getType.toString)
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
    case p: EnumerableProject => {
      val op = ("operator" , "projection")
      val alias = "projection"+p.getId
      val rowType = emitSchema(alias, p.getRowType)
      val child = emit_(p.getInput)
      val childBinding: Binding = child._1
      val childOp = child._2
      //TODO Could also use p.getNamedProjects
      val exprs = p.getProjects
      val exprsJS: JValue = exprs.asScala.map {
        e => emitExpression(e,List(childBinding))
      }

      val json = op ~ ("tupleType", rowType) ~ ("e", exprsJS) ~ ("input" , childOp)
      val binding: Binding = Binding(alias,getFields(p.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case a: EnumerableAggregate => {
      val op = ("operator" , "agg")
      val child = emit_(a.getInput)
      val childBinding: Binding = child._1
      val childOp = child._2

      val groups: List[Integer] = a.getGroupSet.toList.asScala.toList
      val groupsJS: JValue = groups.map {
        g => emitArg(g,List(childBinding))
      }

      val aggs: List[AggregateCall] = a.getAggCallList.asScala.toList
      val aggsJS = aggs.map {
        agg => emitAggExpression(agg,List(childBinding))
      }
      val alias = "agg"+a.getId
      val rowType = emitSchema(alias, a.getRowType)

      val json = op ~ ("tupleType", rowType) ~ ("groups", groupsJS) ~ ("aggs", aggsJS) ~ ("input" , childOp)
      val binding: Binding = Binding(alias,getFields(a.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case j: EnumerableJoin => {
      val op = ("operator" , "join")
      val l = emit_(j.getLeft)
      val leftBinding: Binding = l._1
      val leftChildOp = l._2
      val r = emit_(j.getRight)
      val rightBinding: Binding = r._1
      val rightChildOp = r._2
      val cond = emitExpression(j.getCondition,List(leftBinding,rightBinding))
      val alias = "join"+j.getId
      val rowType = emitSchema(alias, j.getRowType)

      val json = op ~ ("tupleType", rowType) ~ ("cond", cond) ~ ("left" , leftChildOp) ~ ("right" , rightChildOp)
      val binding: Binding = Binding(alias,leftBinding.fields ++ rightBinding.fields)
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case f: EnumerableFilter => {
      val op = ("operator" , "select")
      val child = emit_(f.getInput)
      val childBinding: Binding = child._1
      val childOp = child._2
      val rowType = emitSchema(childBinding.rel, f.getRowType)
      val cond = emitExpression(f.getCondition,List(childBinding))

      val json = op ~ ("tupleType", rowType) ~ ("cond", cond) ~ ("input", childOp)
      val ret: (Binding, JValue) = (childBinding,json)
      ret
    }
    case s : EnumerableTableScan => {
      val op = ("operator" , "scan")
      //TODO Cross-check: 0: schemaName, 1: tableName (?)
      val srcName = s.getTable.getQualifiedName.get(1)
      val rowType = emitSchema(srcName, s.getRowType)

      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName)
      val binding: Binding = Binding(srcName,getFields(s.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case s : PelagoTableScan => {
      val op = ("operator" , "scan")
      //TODO Cross-check: 0: schemaName, 1: tableName (?)
      val srcName  = s.getPelagoRelName //s.getTable.getQualifiedName.get(1)
      val rowType  = emitSchema(srcName, s.getRowType)
      val plugin   = Extraction.decompose(s.getPluginInfo.asScala)
      val linehint = s.getLineHint.longValue

      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName) ~ ("plugin", plugin) ~ ("linehint", linehint)
      val binding: Binding = Binding(srcName,getFields(s.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case s : BindableTableScan => {
      val op = ("operator" , "scan")
      //TODO Cross-check: 0: schemaName, 1: tableName (?)
      val srcName = s.getTable.getQualifiedName.get(1)
      val rowType = emitSchema(srcName, s.getRowType)

      val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName)
      val binding: Binding = Binding(srcName,getFields(s.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case i : EnumerableInterpreter => {
      emit_(i.getInput)
    }
    case sort: EnumerableSort => {
      val op = ("operator" , "sort")
      val child = emit_(sort.getInput)
      val childBinding: Binding = child._1
      val childOp = child._2
      val rowType = emitSchema(childBinding.rel, sort.getRowType)

      val args : JValue =
      sort.getCollation.getFieldCollations.asScala.map {
        col => ("expression", emitArg(col.getFieldIndex,List(childBinding))) ~ ("direction", col.getDirection.shortString)
      }

      val json : JValue = op ~ ("tupleType", rowType) ~ ("args", args) ~ ("input", childOp)
      val ret: (Binding, JValue) = (childBinding,json)
      ret
    }
    case _  => {
      throw new PlanConversionException("Unknown algebraic operator: "+n.getRelTypeName)
    }
  }
}
