package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.prepare.CalcitePrepareImpl;
import org.apache.calcite.rel.RelDeviceTypeTraitDef;

import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Properties;

import org.apache.calcite.test.CalciteAssert;

class PelagoTestConnectionFactory extends CalciteAssert.ConnectionFactory{
  private static final String schemaPath = "../raw-jit-executor/inputs/plans/schema.json";

  private static PelagoTestConnectionFactory instance = null;
  private static Connection connection;

  private PelagoTestConnectionFactory() throws SQLException {
//    Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver");
    Properties info = new Properties();
    connection = DriverManager.getConnection("jdbc:pelago:model=" + schemaPath, info);

    connection.createStatement().executeQuery("explain plan for select * from ssbm_date1000");
  }

  public static PelagoTestConnectionFactory get() throws SQLException {
    if (instance == null) instance = new PelagoTestConnectionFactory();
    return instance;
  }

  @Override public Connection createConnection() {
    return connection;
  }
}


@RunWith(Parameterized.class)
public class PelagoPlannerTest {
  private static final String schemaPath = "../raw-jit-executor/inputs/plans/schema.json";
//    String schemaPath = new java.io.File(".").getCanonicalPath() + "/src/main/resources/schema.json";
  private static Connection connection;

  private static final String queries[] = {
    "select d_year, d_year*8 "
      + "from ssbm_date1000",

    "select sum(lo_revenue), d_year, p_brand1 "
      + "from ssbm_date1000, ssbm_lineorder1000, ssbm_part1000, ssbm_supplier1000 "
      + "where lo_orderdate = d_datekey "
      + "  and lo_partkey = p_partkey "
      + "  and lo_suppkey = s_suppkey "
      + "  and p_category = 'MFGR#12' "
      + "  and s_region = 'AMERICA' "
      + "group by d_year, p_brand1 "
      + "order by d_year, p_brand1",

    "select sum(d_datekey), max(d_datekey) "
      + "from ssbm_date",

    "select sum(lo_revenue), count(*) from "
      + "ssbm_lineorder1000, ssbm_date1000 "
      + "where lo_orderdate = d_datekey "
      + "  and d_year = 1997",

    "select sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder1000, ssbm_customer1000, ssbm_supplier1000 "
      + "where lo_custkey = c_custkey "
      + "  and lo_suppkey = s_suppkey ",

    "select sum(d_year), sum(d_year*8) "
      + "from ssbm_date",

    "select d_year "
      + "from ssbm_date",

    "select sum(lo_revenue) "
      + "from ssbm_lineorder1000, ssbm_date1000 "
      + "where lo_orderdate = d_datekey "
      + "group by d_year",

    "select sum(lo_revenue) "
      + "from ssbm_lineorder1000, ssbm_date1000 "
      + "where lo_orderdate = d_datekey "
      + "group by d_year "
      + "order by d_year desc",

    "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') "
      + "group by d_year, c_nation ",
//      + "order by d_year, c_nation",

    "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') "
      + "group by d_year, c_nation "
      + "order by d_year, c_nation",

    "select count(*) " //sum(lo_revenue - lo_supplycost) as profit " +
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')",

    "select count(*), sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')",

    "select count(*), sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000 "
      + "where lo_custkey = c_custkey "
      + "  and lo_suppkey = s_suppkey "
      + "  and lo_partkey = p_partkey "
      + "  and lo_orderdate = d_datekey "
      + "  and c_region = 'AMERICA' "
      + "  and s_region = 'AMERICA' "
      + "  and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') "
      + "group by d_year, c_nation ",

    // SSB Q1.1
    "select sum(lo_extendedprice*lo_discount) as revenue\n"
      + "from ssbm_lineorder1000, ssbm_date1000\n"
      + "where lo_orderdate = d_datekey\n"
      + " and d_year = 1993\n"
      + " and lo_discount between 1 and 3\n"
      + " and lo_quantity < 25",

    // SSB Q1.2
    "select sum(lo_extendedprice*lo_discount) as revenue\n"
      + "from ssbm_lineorder1000, ssbm_date1000\n"
      + "where lo_orderdate = d_datekey\n"
      + " and d_yearmonthnum = 199401\n"
      + " and lo_discount between 4 and 6\n"
      + " and lo_quantity between 26 and 35",

    // SSB Q1.3
    "select sum(lo_extendedprice*lo_discount) as revenue\n"
      + "from ssbm_lineorder1000, ssbm_date1000\n"
      + "where lo_orderdate = d_datekey\n"
      + " and d_weeknuminyear = 6\n"
      + " and d_year = 1994\n"
      + " and lo_discount between 5 and 7\n"
      + " and lo_quantity between 26 and 35",

    // SSB Q2.1
    "select sum(lo_revenue), d_year, p_brand1\n"
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000\n"
      + " where lo_orderdate = d_datekey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and p_category = 'MFGR#12'\n"
      + " and s_region = 'AMERICA'\n"
      + "group by d_year, p_brand1\n"
      + "order by d_year, p_brand1",

    // SSB Q2.2
    "select sum(lo_revenue) as lo_revenue, d_year, p_brand1\n"
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000\n"
      + "where lo_orderdate = d_datekey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and p_brand1 between 'MFGR#2221' and 'MFGR#2228'\n"
      + " and s_region = 'ASIA' \n"
      + "group by d_year, p_brand1\n"
      + "order by d_year, p_brand1",

    // SSB Q2.3
    "select sum(lo_revenue) as lo_revenue, d_year, p_brand1\n"
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000\n"
      + "where lo_orderdate = d_datekey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and p_brand1 = 'MFGR#2239'\n"
      + " and s_region = 'EUROPE' \n"
      + "group by d_year, p_brand1\n"
      + "order by d_year, p_brand1",

    // SSB Q3.1
    "select c_nation, s_nation, d_year, sum(lo_revenue) as lo_revenue\n"
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and c_region = 'ASIA'\n"
      + " and s_region = 'ASIA'\n"
      + " and d_year >= 1992\n"
      + " and d_year <= 1997 \n"
      + "group by c_nation, s_nation, d_year\n"
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.2
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue\n"
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and c_nation = 'UNITED STATES'\n"
      + " and s_nation = 'UNITED STATES'\n"
      + " and d_year >= 1992\n"
      + " and d_year <= 1997 \n"
      + "group by c_city, s_city, d_year\n"
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.3
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue\n"
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and (c_city='UNITED KI1' or c_city='UNITED KI5')\n"
      + " and (s_city='UNITED KI1' or s_city='UNITED KI5')\n"
      + " and d_year >= 1992\n"
      + " and d_year <= 1997 \n"
      + "group by c_city, s_city, d_year\n"
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.4
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue\n"
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and (c_city='UNITED KI1' or c_city='UNITED KI5')\n"
      + " and (s_city='UNITED KI1' or s_city='UNITED KI5')\n"
      + " and d_yearmonth = 'Dec1997' \n"
      + "group by c_city, s_city, d_year\n"
      + "order by d_year asc, lo_revenue desc",

    // SSB Q4.1
    "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit\n"
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and c_region = 'AMERICA'\n"
      + " and s_region = 'AMERICA'\n"
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') \n"
      + "group by d_year, c_nation\n"
      + "order by d_year, c_nation",

    // SSB Q4.2
    "select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as profit\n"
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and c_region = 'AMERICA'\n"
      + " and s_region = 'AMERICA'\n"
      + " and (d_year = 1997 or d_year = 1998)\n"
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') \n"
      + "group by d_year, s_nation, p_category\n"
      + "order by d_year, s_nation, p_category",

    // SSB Q4.3
    "select d_year, s_city, p_brand1, sum(lo_revenue - lo_supplycost) as profit\n"
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000\n"
      + "where lo_custkey = c_custkey\n"
      + " and lo_suppkey = s_suppkey\n"
      + " and lo_partkey = p_partkey\n"
      + " and lo_orderdate = d_datekey\n"
      + " and c_region = 'AMERICA'\n"
      + " and s_nation = 'UNITED STATES'\n"
      + " and (d_year = 1997 or d_year = 1998)\n"
      + " and p_category = 'MFGR#14' \n"
      + "group by d_year, s_city, p_brand1\n"
      + "order by d_year, s_city, p_brand1"
  };

  private final String sql;

  public PelagoPlannerTest(String sql) {
    this.sql = sql;
  }

  public void plan(String sql) throws SQLException {
    ResultSet resultSet = connection.createStatement().executeQuery("explain plan for " + sql);

    ResultSetMetaData rsmd = resultSet.getMetaData();
    int columnsCount = rsmd.getColumnCount();

    while (resultSet.next()) {
      for (int i = 1; i <= columnsCount; ++i) {
        if (i > 1) System.out.print(",  ");
        System.out.print(resultSet.getString(i) + " " + rsmd.getColumnName(i));
      }
      System.out.println();
    }
  }

  @Test(timeout = 10000)
  public void testParse() throws SQLException {
//    String sql = "select d_year, d_year*8 from ssbm_date1000";
    CalciteAssert.that()
      .with(PelagoTestConnectionFactory.get())
      .query(sql)
      .explainContains("PLAN=PelagoToEnumerableConverter")
      ;

//    plan(sql);
  }

  @BeforeClass
  public static void init() throws SQLException {
//    Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver");

//    Properties info = new Properties();
//    connection = DriverManager.getConnection("jdbc:pelago:model=" + schemaPath, info);
//
//    // warm-up connection
//    connection.createStatement().executeQuery("explain plan for select * from ssbm_date1000");

    // warm-up connection and load classes
    PelagoTestConnectionFactory.get();
  }

  @Parameters(name = "{0}")
  public static Collection<Object[]> data() {
    Collection<Object[]> data = new ArrayList<Object[]>();
    for (String sql: queries) data.add(new Object[]{sql});
    return data;
  }
}