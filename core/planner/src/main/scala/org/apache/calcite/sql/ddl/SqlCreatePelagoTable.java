package org.apache.calcite.sql.ddl;

import org.apache.calcite.sql.parser.SqlParserPos;

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