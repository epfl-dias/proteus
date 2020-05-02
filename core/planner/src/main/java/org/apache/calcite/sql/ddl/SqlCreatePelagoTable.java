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
public class SqlCreatePelagoTable extends SqlCreate
    implements SqlExecutableStatement {
  private final SqlIdentifier name;
  private final SqlNodeList columnList;
  private final SqlNode query;
  private final String jsonPlugin;
  private final String jsonTable;

  private static final SqlOperator OPERATOR =
      new SqlSpecialOperator("CREATE TABLE", SqlKind.CREATE_TABLE);

  /** Creates a SqlCreateTable. */
  SqlCreatePelagoTable(SqlParserPos pos, boolean replace, boolean ifNotExists,
                       SqlIdentifier name, SqlNodeList columnList,
                       SqlNode query, String jsonPlugin, String jsonTable) {
    super(OPERATOR, pos, replace, ifNotExists);
    this.name = Preconditions.checkNotNull(name);
    this.columnList = columnList; // may be null
    this.query = query; // for "CREATE TABLE ... AS query"; may be null
    this.jsonPlugin = jsonPlugin;
    this.jsonTable = jsonTable;
  }

  public List<SqlNode> getOperandList() {
    return ImmutableNullableList.of(name, columnList, query);
  }

  @Override public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword("CREATE");
    writer.keyword("TABLE");
    if (ifNotExists) {
      writer.keyword("IF NOT EXISTS");
    }
    name.unparse(writer, leftPrec, rightPrec);
    if (columnList != null) {
      SqlWriter.Frame frame = writer.startList("(", ")");
      for (SqlNode c : columnList) {
        writer.sep(",");
        c.unparse(writer, 0, 0);
      }
      writer.endList(frame);
    }
    if (query != null) {
      writer.keyword("AS");
      writer.newlineAndIndent();
      query.unparse(writer, 0, 0);
    }
    if (jsonPlugin != null) {
      writer.keyword("JPLUGIN");
      writer.newlineAndIndent();
      writer.identifier(jsonPlugin, false);
    }
  }

  public void execute(CalcitePrepare.Context context) {
    final Pair<CalciteSchema, String> pair =
        SqlDdlPelagoNodes.schema(context, true, name);
    final JavaTypeFactory typeFactory = new JavaTypeFactoryImpl();
    final RelDataType queryRowType;
    if (query != null) {
      // A bit of a hack: pretend it's a view, to get its row type
      final String sql = query.toSqlString(CalciteSqlDialect.DEFAULT).getSql();
      final ViewTableMacro viewTableMacro =
          ViewTable.viewMacro(pair.left.plus(), sql, pair.left.path(null),
              context.getObjectPath(), false);
      final TranslatableTable x = viewTableMacro.apply(ImmutableList.of());
      queryRowType = x.getRowType(typeFactory);

      if (columnList != null
          && queryRowType.getFieldCount() != columnList.size()) {
        throw SqlUtil.newContextException(columnList.getParserPosition(),
            RESOURCE.columnCountMismatch());
      }
    } else {
      queryRowType = null;
    }
    final List<SqlNode> columnList;
    if (this.columnList != null) {
      columnList = this.columnList.getList();
    } else {
      if (queryRowType == null) {
        // FIXME If we enable CREATE TABLE t RAW <json> this will not be the case
        // "CREATE TABLE t" is invalid; because there is no "AS query" we need
        // a list of column names and types, "CREATE TABLE t (INT c)".
          // But it is fine if we define table from JSON text.
        if(jsonTable == null) {
            throw SqlUtil.newContextException(name.getParserPosition(),
                    RESOURCE.createTableRequiresColumnList());
        } else {
            // Parse the JSON and add to schema.
            pair.left.add(pair.right, createTableJSON(jsonTable));
            return;
        }
      }
      columnList = new ArrayList<>();
      for (String name : queryRowType.getFieldNames()) {
        columnList.add(new SqlIdentifier(name, SqlParserPos.ZERO));
      }
    }
    final ImmutableList.Builder<ColumnDef> b = ImmutableList.builder();
    final RelDataTypeFactory.Builder builder = typeFactory.builder();
    final RelDataTypeFactory.Builder storedBuilder = typeFactory.builder();
    final SqlValidator validator = SqlDdlNodes.validator(context, true);
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
          @Override public ColumnStrategy generationStrategy(RelOptTable table,
                                                             int iColumn) {
            return columns.get(iColumn).strategy;
          }

          @Override public RexNode newColumnDefaultValue(RelOptTable table,
                                                         int iColumn, InitializerContext context) {
            final ColumnDef c = columns.get(iColumn);
            if (c.expr != null) {
              return context.convertExpression(c.expr);
            }
            return super.newColumnDefaultValue(table, iColumn, context);
          }
        };
    if (pair.left.plus().getTable(pair.right) != null) {
      // Table exists.
      if (!ifNotExists) {
        // They did not specify IF NOT EXISTS, so give error.
        throw SqlUtil.newContextException(name.getParserPosition(),
            RESOURCE.tableExists(pair.right));
      }
      return;
    }


    // Table does not exist. Create it.
    // If the jsonPlugin is specified, for now it is mandatory
    if(jsonPlugin != null) {
      ObjectMapper mapper = new ObjectMapper();

      try {
        JsonNode root = mapper.readTree(jsonPlugin);

        // pair.left - Calcite Schema, is this correct?
        pair.left.add(pair.right, new PelagoTableFactory().create(pair.left.plus(), pair.right, mapper.convertValue(root, Map.class), rowType));

      } catch (IOException e) {
        e.printStackTrace();
      }

    } else {
      //TODO: what happens if no plugin is specified?
      //throw new Error("plugin is not defined");
      pair.left.add(pair.right,
              new MutableArrayTable(pair.right,
                      RelDataTypeImpl.proto(storedRowType),
                      RelDataTypeImpl.proto(rowType), ief));
    }

    if (query != null) {
      SqlDdlPelagoNodes.populate(name, query, context);
    }
  }

    // TODO As in PelagoSchema, read the json and add it to tables
    // FIXME probably it should be placed in PelagoTable class
    private PelagoTable createTableJSON(String jsonTable) {

        ObjectMapper mapper = new ObjectMapper();
        Map<String, ?> tableDescription = null;

        try {
            tableDescription = mapper.readValue(jsonTable, new TypeReference<Map<String, ?>>() {});
        } catch (IOException e) {
            e.printStackTrace();
        }


        for (Map.Entry<String, ?> e: tableDescription.entrySet()) {
            Map<String, ?> fileEntry = (Map<String, ?>) ((Map<String, ?>) e.getValue()).get("type");
            String fileType = (String) fileEntry.getOrDefault("type", null);
            if (!fileType.equals("bag")) {
                System.err.println("Error in catalog: relation type is expected to be \"bag\", but \"" + fileType + "\" found");
                System.out.println("Ignoring table: " + e.getKey());
                continue;
            }
            Map<String, ?> lineType = (Map<String, ?>) fileEntry.getOrDefault("inner", null);
            if (lineType != null && !lineType.getOrDefault("type", null).equals("record")) lineType = null;
            if (lineType == null) {
                System.err.println("Error in catalog: \"bag\" expected to contain records");
                System.out.println("Ignoring table: " + e.getKey());
                continue;
            }
            Source source = Sources.of(new File((String) ((Map<String, ?>) e.getValue()).get("path")));

            Map<String, ?> plugin = (Map<String, ?>) ((Map<String, ?>) e.getValue()).getOrDefault("plugin", null);
            if (plugin == null) {
                System.err.println("Error in catalog: plugin information not found for table");
                System.out.println("Ignoring table: " + e.getKey());
                continue;
            }

            try {
                // at this point we create the table
                Table table = PelagoTable.create(source, e.getKey(), plugin, lineType, null);
                return (PelagoTable)table;
            } catch (MalformedPlugin malformedPlugin) {
                System.out.println("Error in catalog: " + malformedPlugin.getMessage  ());
                System.out.println("Ignoring table  : " + malformedPlugin.getTableName());
                continue;
            }
        }

        return null;
    }

  /** Column definition. */
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

  /** Abstract base class for implementations of {@link ModifiableTable}. */
  abstract static class AbstractModifiableTable
      extends AbstractTable implements ModifiableTable {
    AbstractModifiableTable(String tableName) {
      super();
    }

    public TableModify toModificationRel(
        RelOptCluster cluster,
        RelOptTable table,
        Prepare.CatalogReader catalogReader,
        RelNode child,
        TableModify.Operation operation,
        List<String> updateColumnList,
        List<RexNode> sourceExpressionList,
        boolean flattened) {
      return LogicalTableModify.create(table, catalogReader, child, operation,
          updateColumnList, sourceExpressionList, flattened);
    }
  }

  /** Table backed by a Java list. */
  public static class MutableArrayTable extends AbstractModifiableTable
      implements Wrapper {
    protected final List rows = new ArrayList();
    private final RelProtoDataType protoStoredRowType;
    private final RelProtoDataType protoRowType;
    private final InitializerExpressionFactory initializerExpressionFactory;

    /** Creates a MutableArrayTable.
     *
     * @param name Name of table within its schema
     * @param protoStoredRowType Prototype of row type of stored columns (all
     *     columns except virtual columns)
     * @param protoRowType Prototype of row type (all columns)
     * @param initializerExpressionFactory How columns are populated
     */
    public MutableArrayTable(String name, RelProtoDataType protoStoredRowType,
        RelProtoDataType protoRowType,
        InitializerExpressionFactory initializerExpressionFactory) {
      super(name);
      this.protoStoredRowType = Preconditions.checkNotNull(protoStoredRowType);
      this.protoRowType = Preconditions.checkNotNull(protoRowType);
      this.initializerExpressionFactory =
          Preconditions.checkNotNull(initializerExpressionFactory);
    }

    public Collection getModifiableCollection() {
      return rows;
    }

    public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
                                        SchemaPlus schema, String tableName) {
      return new AbstractTableQueryable<T>(queryProvider, schema, this,
          tableName) {
        public Enumerator<T> enumerator() {
          //noinspection unchecked
          return (Enumerator<T>) Linq4j.enumerator(rows);
        }
      };
    }

    public Type getElementType() {
      return Object[].class;
    }

    public Expression getExpression(SchemaPlus schema, String tableName,
                                    Class clazz) {
      return Schemas.tableExpression(schema, getElementType(),
          tableName, clazz);
    }

    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      return protoRowType.apply(typeFactory);
    }

    @Override public <C> C unwrap(Class<C> aClass) {
      if (aClass.isInstance(initializerExpressionFactory)) {
        return aClass.cast(initializerExpressionFactory);
      }
      return super.unwrap(aClass);
    }
  }
}

// End SqlCreateTable.java