package org.apache.calcite.sql.ddl;

import ch.epfl.dias.calcite.adapter.pelago.MalformedPlugin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTable;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.linq4j.*;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.logical.LogicalTableModify;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.*;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.schema.impl.AbstractTableQueryable;
import org.apache.calcite.schema.impl.ViewTable;
import org.apache.calcite.schema.impl.ViewTableMacro;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.InitializerContext;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;
import org.apache.calcite.sql2rel.NullInitializerExpressionFactory;
import org.apache.calcite.util.ImmutableNullableList;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Source;
import org.apache.calcite.util.Sources;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;

import static org.apache.calcite.util.Static.RESOURCE;

import org.apache.calcite.sql.*;

/**
 * Parse tree for {@code CREATE TABLE} statement.
 */
public class SqlCreatePelagoTable extends SqlCreateTable {
  private final String jsonPlugin;
  private final String jsonTable;

  /** Creates a SqlCreateTable. */
  SqlCreatePelagoTable(SqlParserPos pos, boolean replace, boolean ifNotExists,
                       SqlIdentifier name, SqlNodeList columnList,
                       SqlNode query, String jsonPlugin, String jsonTable) {
    super(pos, replace, ifNotExists, name, columnList, query);
    this.jsonPlugin = jsonPlugin;
    this.jsonTable = jsonTable;
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    super.unparse(writer, leftPrec, rightPrec);
    if (jsonPlugin != null) {
      writer.keyword("JPLUGIN");
      writer.newlineAndIndent();
      writer.identifier(jsonPlugin, false);
    }
  }

  public String getJsonPlugin() {
    return jsonPlugin;
  }
}

// End SqlCreateTable.java