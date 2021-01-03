package org.apache.calcite.sql.ddl;

import com.google.common.collect.ImmutableList;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.PelagoPrepareImpl;
import org.apache.calcite.prepare.PlannerImpl;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.schema.Schemas;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;
import java.util.Objects;


/**
 * Utilities concerning {@link SqlNode} for DDL.
 */
public class SqlDdlPelagoNodes {
  private SqlDdlPelagoNodes() {}

  /** Creates a CREATE SCHEMA. */
  public static SqlCreateSchema createSchema(SqlParserPos pos, boolean replace,
                                             boolean ifNotExists, SqlIdentifier name) {
    return new SqlCreateSchema(pos, replace, ifNotExists, name);
  }

  /** Creates a CREATE FOREIGN SCHEMA. */
  public static SqlCreateForeignSchema createForeignSchema(SqlParserPos pos,
                                                           boolean replace, boolean ifNotExists, SqlIdentifier name, SqlNode type,
                                                           SqlNode library, SqlNodeList optionList) {
    return new SqlCreateForeignSchema(pos, replace, ifNotExists, name, type,
            library, optionList);
  }

  /** Creates a CREATE TYPE. */
  public static SqlCreateType createType(SqlParserPos pos, boolean replace,
                                         SqlIdentifier name, SqlNodeList attributeList,
                                         SqlDataTypeSpec dataTypeSpec) {
    return new SqlCreateType(pos, replace, name, attributeList, dataTypeSpec);
  }

  /** Creates a CREATE TABLE. */
  public static SqlCreatePelagoTable createPelagoTable(SqlParserPos pos, boolean replace,
                                                 boolean ifNotExists, SqlIdentifier name, SqlNodeList columnList,
                                                 SqlNode query, String jsonPlugin, String jsonTable) {
    return new SqlCreatePelagoTable(pos, replace, ifNotExists, name, columnList,
            query, jsonPlugin, jsonTable);
  }

  /** Creates a CREATE VIEW. */
  public static SqlCreateView createView(SqlParserPos pos, boolean replace,
                                         SqlIdentifier name, SqlNodeList columnList, SqlNode query) {
    return new SqlCreateView(pos, replace, name, columnList, query);
  }

  /** Creates a CREATE MATERIALIZED VIEW. */
  public static SqlCreateMaterializedView createMaterializedView(
          SqlParserPos pos, boolean replace, boolean ifNotExists,
          SqlIdentifier name, SqlNodeList columnList, SqlNode query) {
    return new SqlCreateMaterializedView(pos, replace, ifNotExists, name,
            columnList, query);
  }

  /** Creates a CREATE FUNCTION. */
  public static SqlCreateFunction createFunction(
          SqlParserPos pos, boolean replace, boolean ifNotExists,
          SqlIdentifier name, SqlNode className, SqlNodeList usingList) {
    return new SqlCreateFunction(pos, replace, ifNotExists, name,
            className, usingList);
  }

  /** Creates a DROP [ FOREIGN ] SCHEMA. */
  public static SqlDropSchema dropSchema(SqlParserPos pos, boolean foreign,
                                         boolean ifExists, SqlIdentifier name) {
    return new SqlDropSchema(pos, foreign, ifExists, name);
  }

  /** Creates a DROP TYPE. */
  public static SqlDropType dropType(SqlParserPos pos, boolean ifExists,
                                     SqlIdentifier name) {
    return new SqlDropType(pos, ifExists, name);
  }

  /** Creates a DROP TABLE. */
  public static SqlDropTable dropTable(SqlParserPos pos, boolean ifExists,
                                       SqlIdentifier name) {
    return new SqlDropTable(pos, ifExists, name);
  }

  /** Creates a DROP VIEW. */
  public static SqlDrop dropView(SqlParserPos pos, boolean ifExists,
                                 SqlIdentifier name) {
    return new SqlDropView(pos, ifExists, name);
  }

  /** Creates a DROP MATERIALIZED VIEW. */
  public static SqlDrop dropMaterializedView(SqlParserPos pos,
                                             boolean ifExists, SqlIdentifier name) {
    return new SqlDropMaterializedView(pos, ifExists, name);
  }

  /** Creates a DROP FUNCTION. */
  public static SqlDrop dropFunction(SqlParserPos pos,
                                     boolean ifExists, SqlIdentifier name) {
    return new SqlDropFunction(pos, ifExists, name);
  }

  /** Creates a column declaration. */
  public static SqlNode column(SqlParserPos pos, SqlIdentifier name,
                               SqlDataTypeSpec dataType, SqlNode expression, ColumnStrategy strategy) {
    return new SqlColumnDeclaration(pos, name, dataType, expression, strategy);
  }

  /** Creates a attribute definition. */
  public static SqlNode attribute(SqlParserPos pos, SqlIdentifier name,
                                  SqlDataTypeSpec dataType, SqlNode expression, SqlCollation collation) {
    return new SqlAttributeDefinition(pos, name, dataType, expression, collation);
  }

  /** Creates a CHECK constraint. */
  public static SqlNode check(SqlParserPos pos, SqlIdentifier name,
                              SqlNode expression) {
    return new SqlCheckConstraint(pos, name, expression);
  }

  /** Creates a UNIQUE constraint. */
  public static SqlKeyConstraint unique(SqlParserPos pos, SqlIdentifier name,
                                        SqlNodeList columnList) {
    return new SqlKeyConstraint(pos, name, columnList);
  }

  /** Creates a PRIMARY KEY constraint. */
  public static SqlKeyConstraint primary(SqlParserPos pos, SqlIdentifier name,
                                         SqlNodeList columnList) {
    return new SqlKeyConstraint(pos, name, columnList) {
      @Override public SqlOperator getOperator() {
        return PRIMARY;
      }
    };
  }



  /** Wraps a query to rename its columns. Used by CREATE VIEW and CREATE
   * MATERIALIZED VIEW. */
  static SqlNode renameColumns(SqlNodeList columnList, SqlNode query) {
    if (columnList == null) {
      return query;
    }
    final SqlParserPos p = query.getParserPosition();
    final SqlNodeList selectList =
            new SqlNodeList(ImmutableList.<SqlNode>of(SqlIdentifier.star(p)), p);
    final SqlCall from =
            SqlStdOperatorTable.AS.createCall(p,
                    ImmutableList.<SqlNode>builder()
                            .add(query)
                            .add(new SqlIdentifier("_", p))
                            .addAll(columnList)
                            .build());
    return new SqlSelect(p, null, selectList, from, null, null, null, null,
            null, null, null, null);
  }
//  /** Populates the table called {@code name} by executing {@code query}. */
//  protected static void populate(SqlIdentifier name, SqlNode query,
//                                 CalcitePrepare.Context context) {
//    // Generate, prepare and execute an "INSERT INTO table query" statement.
//    // (It's a bit inefficient that we convert from SqlNode to SQL and back
//    // again.)
//    final FrameworkConfig config = Frameworks.newConfigBuilder()
//            .defaultSchema(SqlDdlPelagoNodes.schema(context, true, name).left.plus())
//            .build();
////    final Planner planner = new PelagoPrepareImpl().createPlanner(context);
//    final Planner planner = Frameworks.getPlanner(config);
//
//    try {
//      final StringBuilder buf = new StringBuilder();
//      final SqlPrettyWriter w =
//          new SqlPrettyWriter(
//              SqlPrettyWriter.config()
//                  .withDialect(CalciteSqlDialect.DEFAULT)
//                  .withAlwaysUseParentheses(false),
//              buf);
//      buf.append("INSERT INTO ");
//      name.unparse(w, 0, 0);
//      buf.append(" ");
//      System.out.println(query);
//      query.unparse(w, 0, 0);
//      final String sql = buf.toString();
//      final SqlNode query1 = planner.parse(sql);
//      System.out.println(query1);
//      final SqlNode query2 = planner.validate(query1);
//      System.out.println(query2);
//      final RelRoot r = planner.rel(query2);
//      System.out.println(RelOptUtil.toString(r.rel, SqlExplainLevel.ALL_ATTRIBUTES));
//      final PreparedStatement prepare = context.getRelRunner().prepare(r.rel);
//      int rowCount = prepare.executeUpdate();
//      Util.discard(rowCount);
//      prepare.close();
//    } catch (SqlParseException | ValidationException
//            | RelConversionException | SQLException e) {
//      throw new RuntimeException(e);
//    }
//  }
//  /** Populates the table called {@code name} by executing {@code query}. */
//  protected static void populate(SqlIdentifier name, SqlNode query,
//                                 CalcitePrepare.Context context) {
//    // Generate, prepare and execute an "INSERT INTO table query" statement.
//    // (It's a bit inefficient that we convert from SqlNode to SQL and back
//    // again.)
//    final FrameworkConfig config = Frameworks.newConfigBuilder()
//            .defaultSchema(context.getRootSchema().plus())
//            .build();
//    final Planner planner = Frameworks.getPlanner(config);
//    try {
//      final StringBuilder buf = new StringBuilder();
//      final SqlPrettyWriter w =
//          new SqlPrettyWriter(
//              SqlPrettyWriter.config()
//                  .withDialect(CalciteSqlDialect.DEFAULT)
//                  .withAlwaysUseParentheses(false),
//              buf);
//      buf.append("INSERT INTO ");
//      name.unparse(w, 0, 0);
//      buf.append(" ");
//      System.out.println(query);
//      query.unparse(w, 0, 0);
//      final String sql = buf.toString();
//      System.out.println(sql);
//      final SqlNode query1 = planner.parse(sql);
//      System.out.println(query1);
//      final SqlNode query2 = planner.validate(query1);
//      System.out.println(query2);
//      final RelRoot r = planner.rel(query2);
//      System.out.println(RelOptUtil.toString(r.rel, SqlExplainLevel.ALL_ATTRIBUTES));
//      final PreparedStatement prepare = context.getRelRunner().prepare(r.rel);
//      int rowCount = prepare.executeUpdate();
//      Util.discard(rowCount);
//      prepare.close();
//    } catch (SqlParseException | ValidationException
//            | RelConversionException | SQLException e) {
//      throw new RuntimeException(e);
//    }
//  }

//
//  /** Populates the table called {@code name} by executing {@code query}. */
//  protected static void populate(SqlIdentifier name, SqlNode query,
//                                 CalcitePrepare.Context context) {
//    // Generate, prepare and execute an "INSERT INTO table query" statement.
//    // (It's a bit inefficient that we convert from SqlNode to SQL and back
//    // again.)
//    final FrameworkConfig config = Frameworks.newConfigBuilder()
//        .defaultSchema(
//            Objects.requireNonNull(
//                Schemas.subSchema(context.getRootSchema(),
//                    context.getDefaultSchemaPath())).plus())
//        .build();
//    final Planner planner = Frameworks.getPlanner(config);
//    try {
//      final StringBuilder buf = new StringBuilder();
//      final SqlPrettyWriter w =
//          new SqlPrettyWriter(
//              SqlPrettyWriter.config()
//                  .withDialect(CalciteSqlDialect.DEFAULT)
//                  .withAlwaysUseParentheses(false),
//              buf);
//      buf.append("INSERT INTO ");
//      name.unparse(w, 0, 0);
//      buf.append(" ");
//      System.out.println(query);
//      query.unparse(w, 0, 0);
//      final String sql = buf.toString();
//      final SqlNode query1 = planner.parse(sql);
//      final SqlNode query2 = planner.validate(query1);
//      System.out.println(query2);
//      final RelRoot r = planner.rel(query2);
//      if (r.rel instanceof Values){
//        System.out.println("Values!!! Calling TX directly");
//      } else {
//        System.out.println(RelOptUtil.toString(r.rel, SqlExplainLevel.ALL_ATTRIBUTES));
//      }
//
//      final PreparedStatement prepare = context.getRelRunner().prepare(r.rel);
//      int rowCount = prepare.executeUpdate();
//      Util.discard(rowCount);
//      prepare.close();
//    } catch (SqlParseException | ValidationException
//        | RelConversionException | SQLException e) {
//      throw new RuntimeException(e);
//    }
//  }


  /** File type for CREATE FUNCTION. */
  public enum FileType {
    FILE,
    JAR,
    ARCHIVE
  }
}

// End SqlDdlNodes.java