package ch.epfl.dias.emitter

import java.io.{PrintWriter, StringWriter}
import java.util

import com.google.common.collect.ImmutableList
import org.apache.calcite.adapter.enumerable._
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.{RelDataType, RelDataTypeField, RelRecordType}
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rex.{RexCall, RexInputRef, RexLiteral, RexNode}
import org.apache.calcite.sql.{SqlBinaryOperator, SqlKind, SqlOperator}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.util.control.Breaks._

object PlanToJSON {

  case class Binding(rel: String, fields: List[RelDataTypeField])

  //-> TODO What about degree of materialization for joins?
  //---> Join behavior can be handled implicitly by executor:
  //-----> Early materialization: Create new OID
  //-----> Late materialization: Preserve OID + rowTypes of join's children instead of the new one

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
    val json : JValue = e match {
      case call: RexCall => emitOp(call.op, call.operands, f)
      case inputRef: RexInputRef => {
        val arg = emitArg(inputRef,f)
        ("arg",arg)
      }
      case lit: RexLiteral => ("v",toJavaString(lit))
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
      case SqlKind.AVG => "avg"
      case SqlKind.COUNT => "cnt"
      case SqlKind.MAX => "max"
      case SqlKind.MIN => "min"
      case SqlKind.SUM => "sum"
      case _ => {
        val msg : String = "unknown aggr. function "+aggExpr.getAggregation.getKind.toString
        throw new PlanConversionException(msg)
      }
    }

    if(aggExpr.getArgList.size() != 1)  {
      val msg : String = "size of aggregate's input expected to be 1 - actually is "+aggExpr.getArgList.size()
      throw new PlanConversionException(msg)
    }
    val e = emitArg(aggExpr.getArgList.get(0),f)
    val json : JValue = ("type", aggExpr.getType.toString) ~ ("op",opType) ~ ("e",e)
    json
  }

  def emitOp(op: SqlOperator, args: ImmutableList[RexNode], f: List[Binding] ) : JValue = op match   {
    case binOp: SqlBinaryOperator => emitBinaryOp(binOp, args, f)
    case _ => throw new PlanConversionException("Unknown operator: "+op.getKind.sql)
  }

  def emitBinaryOp(op: SqlBinaryOperator, args: ImmutableList[RexNode], f: List[Binding]) : JValue =  {

    val opType : String = op.getKind match {
      case SqlKind.GREATER_THAN => "gt"
      case SqlKind.GREATER_THAN_OR_EQUAL => "qe"
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

    val left = emitExpression(args.get(0), f)
    val right = emitExpression(args.get(1), f)
    val json = ("expression",opType) ~ ("left", left) ~ ("right", right)
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

    val json : JObject = ("type",arg.getType.toString)~("rel",rel) ~ ("attr",attr)
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

      val json = op ~ ("type", rowType) ~ ("e", exprsJS) ~ ("child" , childOp)
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

      val json = op ~ ("type", rowType) ~ ("groups", groupsJS) ~ ("aggs", aggsJS) ~ ("child" , childOp)
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

      val json = op ~ ("type", rowType) ~ ("cond", cond) ~ ("left" , leftChildOp) ~ ("right" , rightChildOp)
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

      val json = op ~ ("type", rowType) ~ ("cond", cond) ~ ("input", childOp)
      val ret: (Binding, JValue) = (childBinding,json)
      ret
    }
    case s : EnumerableTableScan => {
      val op = ("operator" , "scan")
      //TODO Cross-check: 0: schemaName, 1: tableName (?)
      val srcName = s.getTable.getQualifiedName.get(1)
      val rowType = emitSchema(srcName, s.getRowType)

      val json : JValue = op~ ("type", rowType) ~ ("name", srcName)
      val binding: Binding = Binding(srcName,getFields(s.getRowType))
      val ret: (Binding, JValue) = (binding,json)
      ret
    }
    case _  => {
      throw new PlanConversionException("Unknown algebraic operator: "+n.getRelTypeName)
    }
  }
}
