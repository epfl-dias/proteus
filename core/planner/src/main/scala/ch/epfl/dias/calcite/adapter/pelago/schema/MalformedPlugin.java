package ch.epfl.dias.calcite.adapter.pelago.schema;

public class MalformedPlugin extends Exception {
  private String tableName;
  public MalformedPlugin(String message, String tableName) {
    super(message);
    this.tableName = tableName;
  }

  public String getTableName(){
    return tableName;
  }
}
