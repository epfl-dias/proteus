package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.test.CalciteAssert;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.repl.Repl;

import java.lang.management.ManagementFactory;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

class PelagoTestConnectionFactory extends CalciteAssert.ConnectionFactory{
  static final boolean isDebug = ManagementFactory.getRuntimeMXBean().getInputArguments().toString().contains("jdwp");
  private static final String schemaPath = "inputs/plans/schema.json";

  private static PelagoTestConnectionFactory instance = null;

  private PelagoTestConnectionFactory() throws SQLException {
    Repl.planfile_$eq ("plan.json");

    Repl.mockfile_$eq ("src/main/resources/mock.csv");
    Repl.isMockRun_$eq(true);
    Repl.printplan_$eq(true);
    Repl.gpudop_$eq(2);
    Repl.cpudop_$eq(24);

    Connection connection = createConnection();
    connection.createStatement().executeQuery("explain plan for select count(d_datekey) from ssbm_date");
    connection.close();
  }

  public static PelagoTestConnectionFactory get() throws SQLException {
    if (instance == null) instance = new PelagoTestConnectionFactory();
    return instance;
  }

  @Override public Connection createConnection() throws SQLException {
    Properties info = new Properties();
    return DriverManager.getConnection("jdbc:proteus:model=" + schemaPath, info);
  }
}
