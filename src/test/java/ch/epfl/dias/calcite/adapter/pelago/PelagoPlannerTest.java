package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.repl.Repl;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

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

    Repl.mockfile_$eq ("src/main/resources/mock.csv");
    Repl.isMockRun_$eq(true);

//    connection.createStatement().execute("ALTER SESSION SET cpuonly = true");

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
  //private static final String schemaPath = "../raw-jit-executor/inputs/plans/schema.json";

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

    "select sum(lo_revenue), count(*) "
      + "from ssbm_lineorder1000, ssbm_date1000 "
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
    "select sum(lo_extendedprice*lo_discount) as revenue "
      + "from ssbm_lineorder, ssbm_date "
      + "where lo_orderdate = d_datekey "
      + " and d_year = 1993 "
      + " and lo_discount between 1 and 3 "
      + " and lo_quantity < 25",

    // SSB Q1.2
    "select sum(lo_extendedprice*lo_discount) as revenue "
      + "from ssbm_lineorder1000, ssbm_date1000 "
      + "where lo_orderdate = d_datekey "
      + " and d_yearmonthnum = 199401 "
      + " and lo_discount between 4 and 6 "
      + " and lo_quantity between 26 and 35",

    // SSB Q1.3
    "select sum(lo_extendedprice*lo_discount) as revenue "
      + "from ssbm_lineorder1000, ssbm_date1000 "
      + "where lo_orderdate = d_datekey "
      + " and d_weeknuminyear = 6 "
      + " and d_year = 1994 "
      + " and lo_discount between 5 and 7 "
      + " and lo_quantity between 26 and 35",

    // SSB Q2.1
    "select sum(lo_revenue), d_year, p_brand1 "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000 "
      + " where lo_orderdate = d_datekey "
      + " and lo_partkey = p_partkey "
      + " and lo_suppkey = s_suppkey "
      + " and p_category = 'MFGR#12' "
      + " and s_region = 'AMERICA' "
      + "group by d_year, p_brand1 "
      + "order by d_year, p_brand1",

    // SSB Q2.2
    "select sum(lo_revenue) as lo_revenue, d_year, p_brand1 "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000 "
      + "where lo_orderdate = d_datekey "
      + " and lo_partkey = p_partkey "
      + " and lo_suppkey = s_suppkey "
      + " and p_brand1 between 'MFGR#2221' and 'MFGR#2228' "
      + " and s_region = 'ASIA'  "
      + "group by d_year, p_brand1 "
      + "order by d_year, p_brand1",

    // SSB Q2.3
    "select sum(lo_revenue) as lo_revenue, d_year, p_brand1 "
      + "from ssbm_lineorder1000, ssbm_date1000, ssbm_part1000, ssbm_supplier1000 "
      + "where lo_orderdate = d_datekey "
      + " and lo_partkey = p_partkey "
      + " and lo_suppkey = s_suppkey "
      + " and p_brand1 = 'MFGR#2239' "
      + " and s_region = 'EUROPE'  "
      + "group by d_year, p_brand1 "
      + "order by d_year, p_brand1",

    // SSB Q3.1
    "select c_nation, s_nation, d_year, sum(lo_revenue) as lo_revenue "
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'ASIA' "
      + " and s_region = 'ASIA' "
      + " and d_year >= 1992 "
      + " and d_year <= 1997  "
      + "group by c_nation, s_nation, d_year "
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.2
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue "
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_orderdate = d_datekey "
      + " and c_nation = 'UNITED STATES' "
      + " and s_nation = 'UNITED STATES' "
      + " and d_year >= 1992 "
      + " and d_year <= 1997  "
      + "group by c_city, s_city, d_year "
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.3
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue "
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_orderdate = d_datekey "
      + " and (c_city='UNITED KI1' or c_city='UNITED KI5') "
      + " and (s_city='UNITED KI1' or s_city='UNITED KI5') "
      + " and d_year >= 1992 "
      + " and d_year <= 1997  "
      + "group by c_city, s_city, d_year "
      + "order by d_year asc, lo_revenue desc",

    // SSB Q3.4
    "select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue "
      + "from ssbm_customer1000, ssbm_lineorder1000, ssbm_supplier1000, ssbm_date1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_orderdate = d_datekey "
      + " and (c_city='UNITED KI1' or c_city='UNITED KI5') "
      + " and (s_city='UNITED KI1' or s_city='UNITED KI5') "
      + " and d_yearmonth = 'Dec1997'  "
      + "group by c_city, s_city, d_year "
      + "order by d_year asc, lo_revenue desc",

    // SSB Q4.1
    "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')  "
      + "group by d_year, c_nation "
      + "order by d_year, c_nation",

    // SSB Q4.2
    "select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (d_year = 1997 or d_year = 1998) "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')  "
      + "group by d_year, s_nation, p_category "
      + "order by d_year, s_nation, p_category",

    // SSB Q4.3
    "select d_year, s_city, p_brand1, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_date1000, ssbm_customer1000, ssbm_supplier1000, ssbm_part1000, ssbm_lineorder1000 "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_nation = 'UNITED STATES' "
      + " and (d_year = 1997 or d_year = 1998) "
      + " and p_category = 'MFGR#14'  "
      + "group by d_year, s_city, p_brand1 "
      + "order by d_year, s_city, p_brand1",

     // "case" query
    "select case when d_year < 15 then case when d_year > 0 then 2 else 0 end else 1 end "
      + "from ssbm_date",

    // unnest query
    "select age, age2 from employeesnum e, unnest(e.children) as c",

    // unnest + group by query
    "select sum(age), age2 from employeesnum e, unnest(e.children) as c group by age2",

    // unnest + group by query
    "select count(A1) from A",

    "select avg(A1) from A",

    "select avg(A1 + 1) from A",

    "select avg(cast(A1 as double)) from A",

    "select avg(cast(A1 + 1.5 as double)) from A",

    "select avg(A1 + 1.5) from A",

    "select avg(A1 + 0.0) from A",

    "select count(lo_orderdate) from ssbm_lineorder_csv",

     //FIXME: should add iris dataset and enable the following test
//    "SELECT AVG(sepal_len) AS avg_sepal_len, AVG(sepal_wid) AS avg_sepal_wid, (CASE WHEN (aaa.P1<aaa.P2) AND (aaa.P1<aaa.P3) THEN 1 WHEN (aaa.P2<aaa.P3) THEN 2 ELSE 3 END) AS `member` FROM (SELECT sepal_len, sepal_wid, ((sepal_len-7.48677641861141)*(sepal_len-7.48677641861141)+(sepal_wid-4.21831973535009)*(sepal_wid-4.21831973535009)) AS P1, ((sepal_len-4.38052375022089)*(sepal_len-4.38052375022089)+(sepal_wid-4.07804339565337)*(sepal_wid-4.07804339565337)) AS P2, ((sepal_len-5.6342051673797)*(sepal_len-5.6342051673797)+(sepal_wid-4.20594438808039)*(sepal_wid-4.20594438808039)) AS P3 FROM iris) aaa GROUP BY (CASE WHEN (aaa.P1<aaa.P2) AND (aaa.P1<aaa.P3) THEN 1 WHEN (aaa.P2<aaa.P3) THEN 2 ELSE 3 END)",

//    // nest
//    "select d_yearmonthnum, collect(d_datekey), collect(1) from ssbm_date group by d_yearmonthnum",
//
//
//    "select count(*) "
//      + "from ( "
//      + " select d_yearmonthnum, collect(d_datekey) as x, collect(1) as y from ssbm_date group by d_yearmonthnum "
//      + ") as c, unnest(c.x) "
//      + "where d_yearmonthnum > 199810 ",

      "select sum(l_orderkey) from tpch1_lineitem",

      //TPC-H 1
      "select "
        + "  l_returnflag, "
        + "  l_linestatus, "
        + "  sum(l_quantity) as sum_qty, "
        + "  sum(l_extendedprice) as sum_base_price, "
        + "  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
        + "  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
        + "  avg(l_quantity) as avg_qty, "
        + "  avg(l_extendedprice) as avg_price, "
        + "  avg(l_discount) as avg_disc, "
        + "  count(*) as count_order "
        + "from "
        + "  tpch1_lineitem "
        + "where "
        + "  l_shipdate <= date '1998-12-01' - interval '90' day "
        + "group by "
        + "  l_returnflag, "
        + "  l_linestatus "
        + "order by "
        + "  l_returnflag, "
        + "  l_linestatus ",
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

  @Test (timeout = 10000)
  public void testParse() throws SQLException {
//    String sql = "select d_year, d_year*8 from ssbm_date1000";
    CalciteAssert.that()
      .with(PelagoTestConnectionFactory.get())
      .query(sql)
      .explainContains("PLAN=PelagoToEnumerableConverter")
      .runs()
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
