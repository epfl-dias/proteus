package ch.epfl.dias.calcite.adapter.pelago;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.sql.SQLException;
import java.time.Duration;
import java.util.stream.Stream;

import org.apache.calcite.test.CalciteAssert;

public class PelagoPlannerTest {
  private static final String queries[] = {
    "select d_year, d_year*8 "
      + "from ssbm_date",
    "select sum(lo_revenue), d_year, p_brand1 "
      + "from ssbm_date, ssbm_lineorder, ssbm_part, ssbm_supplier "
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
      + "from ssbm_lineorder, ssbm_date "
      + "where lo_orderdate = d_datekey "
      + "  and d_year = 1997",

    "select sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder, ssbm_customer, ssbm_supplier "
      + "where lo_custkey = c_custkey "
      + "  and lo_suppkey = s_suppkey ",

    "select sum(d_year), sum(d_year*8) "
      + "from ssbm_date",

    "select d_year "
      + "from ssbm_date",

    "select sum(lo_revenue) "
      + "from ssbm_lineorder, ssbm_date "
      + "where lo_orderdate = d_datekey "
      + "group by d_year",

    "select sum(lo_revenue) "
      + "from ssbm_lineorder, ssbm_date "
      + "where lo_orderdate = d_datekey "
      + "group by d_year "
      + "order by d_year desc",

    "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part "
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
      + "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part "
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
      + "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')",

    "select count(*), sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part "
      + "where lo_custkey = c_custkey "
      + " and lo_suppkey = s_suppkey "
      + " and lo_partkey = p_partkey "
      + " and lo_orderdate = d_datekey "
      + " and c_region = 'AMERICA' "
      + " and s_region = 'AMERICA' "
      + " and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')",

    "select count(*), sum(lo_revenue - lo_supplycost) as profit "
      + "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part "
      + "where lo_custkey = c_custkey "
      + "  and lo_suppkey = s_suppkey "
      + "  and lo_partkey = p_partkey "
      + "  and lo_orderdate = d_datekey "
      + "  and c_region = 'AMERICA' "
      + "  and s_region = 'AMERICA' "
      + "  and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') "
      + "group by d_year, c_nation ",

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
  };


  @ParameterizedTest
  @MethodSource("data")
  public void testParse(String sql) {
    Assertions.assertTimeoutPreemptively(Duration.ofSeconds(
        (PelagoTestConnectionFactory.isDebug) ? (Long.MAX_VALUE / 1000) : 10),
      () -> {
        CalciteAssert.that()
          .with(PelagoTestConnectionFactory.get())
          .query(sql)
          .explainContains("PLAN=PelagoToEnumerableConverter")
          .runs()
        ;
    });
  }

  @BeforeAll
  public static void init() throws SQLException {
    // warm-up connection and load classes
    PelagoTestConnectionFactory.get();
  }

  public static Stream<String> data() {
    return Stream.of(queries);
  }
}
