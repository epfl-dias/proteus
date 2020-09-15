package org.apache.calcite.server;

import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.sql2rel.InitializerExpressionFactory;

public class PelagoMutableArrayTable extends MutableArrayTable {
  /**
   * Creates a MutableArrayTable.
   *
   * @param name                         Name of table within its schema
   * @param protoStoredRowType           Prototype of row type of stored columns (all
   *                                     columns except virtual columns)
   * @param protoRowType                 Prototype of row type (all columns)
   * @param initializerExpressionFactory How columns are populated
   */
  public PelagoMutableArrayTable(String name, RelProtoDataType protoStoredRowType, RelProtoDataType protoRowType, InitializerExpressionFactory initializerExpressionFactory) {
    super(name, protoStoredRowType, protoRowType, initializerExpressionFactory);
  }
}
