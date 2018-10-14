package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.test.CalciteAssert;

import ch.epfl.dias.repl.Repl;
import ch.epfl.dias.repl.Repl$;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

class PelagoTestConnectionFactory extends CalciteAssert.ConnectionFactory{
  private static final String schemaPath = "../raw-jit-executor/inputs/plans/schema.json";
//  private static final String schemaPath = "inputs/plans/schema.json";

  private static PelagoTestConnectionFactory instance = null;
  private static Connection connection;

  private PelagoTestConnectionFactory() throws SQLException {
//    Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver");
    Properties info = new Properties();
    connection = DriverManager.getConnection("jdbc:pelago:model=" + schemaPath, info);

    Repl.planfile_$eq ("plan.json");

    Repl.mockfile_$eq ("src/main/resources/mock.csv");
    Repl.isMockRun_$eq(true);

//    Repl.isMockRun_$eq(false);
//
//    Repl.timings_$eq(false);
//
//    Repl.cpuonly_$eq(false);
//    Repl.cpudop_$eq(24);
//    Repl.gpudop_$eq(2);

//    connection.createStatement().execute("ALTER SESSION SET cpuonly = true");
//    connection.createStatement().execute("ALTER SESSION SET cpudop = 32");
//    connection.createStatement().execute("ALTER SESSION SET timings = CSV");
//    connection.createStatement().execute("ALTER SESSION SET timings = off");
//    connection.createStatement().execute("ALTER SESSION SET timings = TEXT");
//    connection.createStatement().execute("ALTER SESSION SET timings = on");

    connection.createStatement().executeQuery("explain plan for select count(d_datekey) from ssbm_date");
  }

  public static PelagoTestConnectionFactory get() throws SQLException {
    if (instance == null) instance = new PelagoTestConnectionFactory();
    return instance;
  }

  @Override public Connection createConnection() {
    return connection;
  }
}
