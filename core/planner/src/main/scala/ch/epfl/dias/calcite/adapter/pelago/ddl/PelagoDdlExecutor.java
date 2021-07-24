package ch.epfl.dias.calcite.adapter.pelago.ddl;

import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTableFactory;
import ch.epfl.dias.repl.Repl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.jdbc.ContextSqlValidator;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.schema.TranslatableTable;
import org.apache.calcite.schema.impl.ViewTable;
import org.apache.calcite.schema.impl.ViewTableMacro;
import org.apache.calcite.server.DdlExecutor;
import org.apache.calcite.server.ServerDdlExecutor;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.ddl.SqlColumnDeclaration;
import org.apache.calcite.sql.ddl.SqlCreatePelagoTable;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.parser.SqlAbstractParserImpl;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParserImplFactory;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.InitializerContext;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;
import org.apache.calcite.sql2rel.NullInitializerExpressionFactory;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;

import java.io.Reader;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.calcite.util.Static.RESOURCE;

public class PelagoDdlExecutor extends ServerDdlExecutor {
  public static final PelagoDdlExecutor INSTANCE = new PelagoDdlExecutor();

  @SuppressWarnings("unused") // Used by Driver, through reflection, to setup the parser
  public static final SqlParserImplFactory PARSER_FACTORY =
      new SqlParserImplFactory() {
        @Override
        public SqlAbstractParserImpl getParser(Reader stream) {
          return ch.epfl.dias.calcite.sql.parser.ddl.SqlDdlParserImpl.FACTORY.getParser(stream);
        }

        @Override
        public DdlExecutor getDdlExecutor() {
          return PelagoDdlExecutor.INSTANCE;
        }
      };


  /**
   * Returns the schema in which to create an object.
   */
  static Pair<CalciteSchema, String> schema(CalcitePrepare.Context context,
                                            boolean mutable, SqlIdentifier id) {
    final String name;
    final List<String> path;
    if (id.isSimple()) {
      path = context.getDefaultSchemaPath();
      name = id.getSimple();
    } else {
      path = Util.skipLast(id.names);
      name = Util.last(id.names);
    }
    CalciteSchema schema = mutable ? context.getMutableRootSchema()
        : context.getRootSchema();
    for (String p : path) {
      schema = schema.getSubSchema(p, true);
    }
    return Pair.of(schema, name);
  }

  /**
   * Column definition.
   */
  private static class ColumnDef {
    final SqlNode expr;
    final RelDataType type;
    final ColumnStrategy strategy;

    private ColumnDef(SqlNode expr, RelDataType type,
                      ColumnStrategy strategy) {
      this.expr = expr;
      this.type = type;
      this.strategy = Preconditions.checkNotNull(strategy);
      Preconditions.checkArgument(
          strategy == ColumnStrategy.NULLABLE
              || strategy == ColumnStrategy.NOT_NULLABLE
              || expr != null);
    }

    static ColumnDef of(SqlNode expr, RelDataType type,
                        ColumnStrategy strategy) {
      return new ColumnDef(expr, type, strategy);
    }
  }

  /**
   * Returns the SqlValidator with the given {@code context} schema
   * and type factory.
   */
  static SqlValidator validator(CalcitePrepare.Context context,
                                boolean mutable) {
    return new ContextSqlValidator(context, mutable);
  }

  /**
   * Executes a {@code CREATE TABLE} command.
   */
  public void execute(SqlCreateTable create,
                      CalcitePrepare.Context context) {
    final Pair<CalciteSchema, String> pair =
        schema(context, true, create.name);
    final JavaTypeFactory typeFactory = context.getTypeFactory();
    final RelDataType queryRowType;
    if (create.query != null) {
      // A bit of a hack: pretend it's a view, to get its row type
      final String sql =
          create.query.toSqlString(CalciteSqlDialect.DEFAULT).getSql();
      final ViewTableMacro viewTableMacro =
          ViewTable.viewMacro(pair.left.plus(), sql, pair.left.path(null),
              context.getObjectPath(), false);
      final TranslatableTable x = viewTableMacro.apply(ImmutableList.of());
      queryRowType = x.getRowType(typeFactory);

      if (create.columnList != null
          && queryRowType.getFieldCount() != create.columnList.size()) {
        throw SqlUtil.newContextException(
            create.columnList.getParserPosition(),
            RESOURCE.columnCountMismatch());
      }
    } else {
      queryRowType = null;
    }
    final List<SqlNode> columnList;
    if (create.columnList != null) {
      columnList = create.columnList.getList();
    } else {
      if (queryRowType == null) {
        // "CREATE TABLE t" is invalid; because there is no "AS query" we need
        // a list of column names and types, "CREATE TABLE t (INT c)".
        throw SqlUtil.newContextException(create.name.getParserPosition(),
            RESOURCE.createTableRequiresColumnList());
      }
      columnList = new ArrayList<>();
      for (String name : queryRowType.getFieldNames()) {
        columnList.add(new SqlIdentifier(name, SqlParserPos.ZERO));
      }
    }
    final ImmutableList.Builder<ColumnDef> b = ImmutableList.builder();
    final RelDataTypeFactory.Builder builder = typeFactory.builder();
    final RelDataTypeFactory.Builder storedBuilder = typeFactory.builder();
    // REVIEW 2019-08-19 Danny Chan: Should we implement the
    // #validate(SqlValidator) to get the SqlValidator instance?
    final SqlValidator validator = validator(context, true);
    for (Ord<SqlNode> c : Ord.zip(columnList)) {
      if (c.e instanceof SqlColumnDeclaration) {
        final SqlColumnDeclaration d = (SqlColumnDeclaration) c.e;
        final RelDataType type = d.dataType.deriveType(validator, true);
        builder.add(d.name.getSimple(), type);
        if (d.strategy != ColumnStrategy.VIRTUAL) {
          storedBuilder.add(d.name.getSimple(), type);
        }
        b.add(ColumnDef.of(d.expression, type, d.strategy));
      } else if (c.e instanceof SqlIdentifier) {
        final SqlIdentifier id = (SqlIdentifier) c.e;
        if (queryRowType == null) {
          throw SqlUtil.newContextException(id.getParserPosition(),
              RESOURCE.createTableRequiresColumnTypes(id.getSimple()));
        }
        final RelDataTypeField f = queryRowType.getFieldList().get(c.i);
        final ColumnStrategy strategy = f.getType().isNullable()
            ? ColumnStrategy.NULLABLE
            : ColumnStrategy.NOT_NULLABLE;
        b.add(ColumnDef.of(c.e, f.getType(), strategy));
        builder.add(id.getSimple(), f.getType());
        storedBuilder.add(id.getSimple(), f.getType());
      } else {
        throw new AssertionError(c.e.getClass());
      }
    }
    final RelDataType rowType = builder.build();
    final RelDataType storedRowType = storedBuilder.build();
    final List<ColumnDef> columns = b.build();
    final InitializerExpressionFactory ief =
        new NullInitializerExpressionFactory() {
          @Override
          public ColumnStrategy generationStrategy(RelOptTable table,
                                                   int iColumn) {
            return columns.get(iColumn).strategy;
          }

          @Override
          public RexNode newColumnDefaultValue(RelOptTable table,
                                               int iColumn, InitializerContext context) {
            final ColumnDef c = columns.get(iColumn);
            if (c.expr != null) {
              // REVIEW Danny 2019-10-09: Should we support validation for DDL nodes?
              final SqlNode validated = context.validateExpression(storedRowType, c.expr);
              // The explicit specified type should have the same nullability
              // with the column expression inferred type,
              // actually they should be exactly the same.
              return context.convertExpression(validated);
            }
            return super.newColumnDefaultValue(table, iColumn, context);
          }
        };
    if (pair.left.plus().getTable(pair.right) != null) {
      // Table exists.
      if (!create.ifNotExists) {
        // They did not specify IF NOT EXISTS, so give error.
        throw SqlUtil.newContextException(create.name.getParserPosition(),
            RESOURCE.tableExists(pair.right));
      }
      return;
    }
    // Table does not exist. Create it.
    Map<String, Object> plugin;
    String jsonPlugin = ((SqlCreatePelagoTable) create).getJsonPlugin();
    if (jsonPlugin != null) {
      var mapper = new ObjectMapper();
      JsonNode root;
      try {
        root = mapper.readTree(jsonPlugin);
      } catch (JsonProcessingException e) {
        throw new RuntimeException("Malformed plugin info", e);
      }
      plugin = mapper.convertValue(root, Map.class);
    } else {
      // FIXME: modify file path for now. The correct fix would befor proteus not to use /dev/shm by default
      //  and clotho to handle it
      plugin = Map.of("file", "/dev/shm/" + pair.right + "/" + pair.right, "plugin", Map.of("type", "block", "linehint", 0));
    }

    pair.left.add(pair.right,
        new PelagoTableFactory().create(pair.left.plus(), pair.right, plugin, rowType)
    );

    if (create.query != null) {
//      populate(create.name, create.query, context);
    }
  }


  /**
   * Populates the table called {@code name} by executing {@code query}.
   */
  static void populate(SqlIdentifier name, SqlNode query,
                       CalcitePrepare.Context context) {
    // Generate, prepare and execute an "INSERT INTO table query" statement.
    // (It's a bit inefficient that we convert from SqlNode to SQL and back
    // again.)c
    final FrameworkConfig config = Frameworks.newConfigBuilder()
        .defaultSchema(schema(context, true, name).left.plus())
        .build();
    final Planner planner = Frameworks.getPlanner(config);
    try {
      final StringBuilder buf = new StringBuilder();
      final SqlWriterConfig writerConfig =
          SqlPrettyWriter.config().withAlwaysUseParentheses(false);
      final SqlPrettyWriter w = new SqlPrettyWriter(writerConfig, buf);
      buf.append("INSERT INTO ");
      name.unparse(w, 0, 0);
      buf.append(' ');
      query.unparse(w, 0, 0);
      final String sql = buf.toString();
      final SqlNode query1 = planner.parse(sql);
      final SqlNode query2 = planner.validate(query1);
      final RelRoot r = planner.rel(query2);
      final PreparedStatement prepare = context.getRelRunner().prepare(r.rel);
      int rowCount = prepare.executeUpdate();
      Util.discard(rowCount);
      prepare.close();
    } catch (SqlParseException | ValidationException
        | RelConversionException | SQLException e) {
      throw new RuntimeException(e);
    }
  }

  public void execute(SqlSetOption node, CalcitePrepare.Context context) {
    switch (node.getName().names.get(0)) {
      case "compute_units":
      case "compute":
      case "hwmode":
      case "cu": {
        SqlNode value = node.getValue();
        String option;
        if (value instanceof SqlIdentifier) {
          SqlIdentifier id = (SqlIdentifier) value;
          option = id.toString();
        } else if (value instanceof SqlLiteral) {
          SqlLiteral lit = (SqlLiteral) value;
          option = lit.toValue();
        } else {
          throw new UnsupportedOperationException();
        }
        option = option.toLowerCase();
        switch (option) {
          case "cpu":
          case "cpuonly": {
            Repl.setCpuonly();
            return;
          }
          case "gpu":
          case "gpuonly": {
            Repl.setGpuonly();
            return;
          }
          case "all":
          case "hybrid": {
            Repl.setHybrid();
            return;
          }
          default:
            throw new UnsupportedOperationException();
        }
      }
      case "cpudop": {
        SqlNode value = node.getValue();
        if (value instanceof SqlLiteral) {
          SqlLiteral lit = (SqlLiteral) value;
          int new_cpudop = lit.intValue(true);
          if (new_cpudop > 0) {
            Repl.cpudop_$eq(new_cpudop);
            return;
          }
        }
        throw new UnsupportedOperationException();
      }
      case "gpudop": {
        SqlNode value = node.getValue();
        if (value instanceof SqlLiteral) {
          SqlLiteral lit = (SqlLiteral) value;
          int new_gpudop = lit.intValue(true);
          if (new_gpudop >= 0) {
            Repl.gpudop_$eq(new_gpudop);
            return;
          }
        }
        throw new UnsupportedOperationException();
      }
      case "timings": {
        SqlNode value = node.getValue();
        String option;
        if (value instanceof SqlIdentifier) {
          SqlIdentifier id = (SqlIdentifier) value;
          option = id.toString();
        } else if (value instanceof SqlLiteral) {
          SqlLiteral lit = (SqlLiteral) value;
          option = lit.toValue();
        } else {
          throw new UnsupportedOperationException();
        }
        option = option.toLowerCase();
        switch (option) {
          case "csv": {
            Repl.timingscsv_$eq(true);
            Repl.timings_$eq(true);
            return;
          }
          case "text":
          case "on": {
            Repl.timingscsv_$eq(false);
            Repl.timings_$eq(true);
            return;
          }
          case "off": {
            Repl.timings_$eq(false);
            return;
          }
        }
        throw new UnsupportedOperationException();
      }
    }
  }
}
