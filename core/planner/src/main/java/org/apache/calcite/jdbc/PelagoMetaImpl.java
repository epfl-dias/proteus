package org.apache.calcite.jdbc;

import org.apache.calcite.avatica.AvaticaConnection;

public class PelagoMetaImpl extends CalciteMetaImpl{
  public PelagoMetaImpl(AvaticaConnection connection) {
    super((CalciteConnectionImpl) connection);
  }

  @Override
  public void commit(ConnectionHandle ch) {
    System.err.println("Commit request ignored");
  }

  @Override
  public void rollback(ConnectionHandle ch) {
    System.err.println("Rollback request ignored");
  }
}
