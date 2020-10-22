package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.test.CalciteAssert;
import org.apache.calcite.test.Matchers;
import org.apache.calcite.util.TestUtil;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.repl.Repl;
import org.junit.Assert;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicNode;
import org.junit.jupiter.api.TestFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

import static java.time.Duration.ofSeconds;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

public class PelagoQueryTest {
  private static final Logger logger = LoggerFactory.getLogger(PelagoQueryTest.class);

  @BeforeAll
  public static void init() throws SQLException {
    // warm-up connection and load classes
    PelagoTestConnectionFactory.get();
  }

  @AfterAll
  public static void deinit() throws SQLException {
    // Print timing results
    var conn =
        PelagoTestConnectionFactory.get()
            .createConnection();
//
//    conn
//        .createStatement()
////        .executeQuery("")
//        .execute("CREATE TABLE t (i INTEGER)");
//
//    conn
//        .createStatement()
////        .executeQuery("")
//        .execute("INSERT INTO t VALUES (5)");
//    {
//
//      var stmt = conn.prepareStatement("SELECT d_datekey FROM ssbm_date");
//
//      for (int j = 0 ; j < 5 ; ++j) {
//        stmt.execute();
//        ResultSet rs = stmt.getResultSet();
//
//
//        ResultSetMetaData rsmd = rs.getMetaData();
//        int columnsN = rsmd.getColumnCount();
//        while (rs.next()){
//          // Keep lines in a string and print whole lines each time,
//          // otherwise in junit prints each part a separate info line.
//          ImmutableList.Builder builder = ImmutableList.<String>builder();
//          for (int i = 1 ; i <= columnsN ; ++i){
//            String columnValue = rs.getString(i);
//            builder.add(rsmd.getColumnName(i) + "=" + columnValue);
//          }
//          System.out.println(String.join("; ", builder.build()));
//        }
//      }
//    }
//
//    conn.createStatement()
////        .executeQuery("")
////        .execute("INSERT INTO t SELECT d_datekey FROM ssbm_date");
//        .executeQuery("SELECT d_datekey FROM ssbm_date");
//
//    conn.createStatement()
////        .executeQuery("")
////        .execute("INSERT INTO t SELECT d_datekey FROM ssbm_date");
//        .executeQuery("SELECT d_datekey FROM ssbm_date");
//
//
//
//
//    conn.createStatement()
////        .executeQuery("")
////        .execute("INSERT INTO t SELECT d_datekey FROM ssbm_date");
//        .execute("CREATE TABLE D AS SELECT d_datekey FROM ssbm_date");
////        .execute("CREATE TABLE D AS VALUES (1, 'a'), (2, 'bc')");//SELECT d_datekey FROM ssbm_date");
//
//    conn.createStatement()
////        .executeQuery("")
////        .execute("INSERT INTO t SELECT d_datekey FROM ssbm_date");
//        .execute("INSERT INTO D SELECT d_datekey FROM ssbm_date");
////        .execute("CREATE TABLE D AS VALUES (1, 'a'), (2, 'bc')");//SELECT d_datekey FROM ssbm_date");
//    ResultSet rs = PelagoTestConnectionFactory.get()
//      .createConnection()
//      .createStatement()
//      .executeQuery("SELECT * FROM SessionTimings");
//
//    ResultSetMetaData rsmd = rs.getMetaData();
//    int columnsN = rsmd.getColumnCount();
//    while (rs.next()){
//      // Keep lines in a string and print whole lines each time,
//      // otherwise in junit prints each part a separate info line.
//      ImmutableList.Builder builder = ImmutableList.<String>builder();
//      for (int i = 1 ; i <= columnsN ; ++i){
//        String columnValue = rs.getString(i);
//        builder.add(rsmd.getColumnName(i) + "=" + columnValue);
//      }
//      System.out.println(String.join("; ", builder.build()));
//    }
  }

  private CalciteAssert.AssertQuery testParseAndExecute(String sql) throws SQLException {
    return CalciteAssert.that()
      .with(PelagoTestConnectionFactory.get())
      .query(sql)
      ;
  }

  public static String getModeAsString(){
    if (Repl.isHybrid()) return "hyb";
    if (Repl.isGpuonly()) return "gpu";
    if (Repl.isCpuonly()) return "cpu";
    assert(false);
    return "";
  }

  private static Path validationFile(String queryFile){
    final Path p = Paths.get(queryFile + "." + getModeAsString() + ".plan");
    if (Files.exists(p) && Files.isRegularFile(p)) return p;
    return null;
  }

  public void testQueryOrdered(String sql, Path resultFile) throws SQLException, IOException {
    CalciteAssert.AssertQuery q = testParseAndExecute(sql);

    String pf = Repl.planfile();
    String explainContains = "PLAN=PelagoToEnumerableConverter";
    if (resultFile != null) {
      final String suffix = ".resultset";
      String f = resultFile.toString();
      assert(f.endsWith(suffix));
      final String filename = f.substring(0, f.length() - suffix.length());
      final Path p = validationFile(filename);
      if (p != null){
        String plan = new String(Files.readAllBytes(p)).trim();
        plan = plan.replaceAll("\\s*--[^\n]*", "");
        explainContains = "PLAN=" + plan;
//      } else {
//        q = q.explainMatches("EXCLUDING ATTRIBUTES ", (rs) -> {
//          ResultSetMetaData metaData = null;
//          try {
//            rs.next();
//            String s = rs.getString(1);
//            System.out.println(resultFile);
//            System.out.println(p);
//            Files.write(p, ImmutableList.of(s));
//          } catch (SQLException e) {
//            e.printStackTrace();
//          } catch (IOException e) {
//            e.printStackTrace();
//          }
//        });
      }
      {
        String fname = resultFile.getFileName().toString();
        String qname = fname.substring(0, fname.length() - suffix.length());
        Path target = Paths.get(resultFile.getParent().toString(), "current", getModeAsString(), qname + ".json");
        target.getParent().toFile().mkdirs();
        Repl.planfile_$eq(target.toString());
        System.out.println(Repl.planfile());
      }
    }
    assert(explainContains.startsWith("PLAN=PelagoToEnumerableConverter"));

    q = q.explainMatches("EXCLUDING ATTRIBUTES ", CalciteAssert.checkResultContains(explainContains));
    Repl.planfile_$eq(pf);

    if (resultFile != null && !Repl.isMockRun()){
      List<String> lines = Files.readAllLines(resultFile);
      q.returnsOrdered(lines.toArray(new String[lines.size()]));
    } else {
      if (resultFile != null) logger.debug("Result set not verified (reason: running in mock mode)");
      if (!Repl.isMockRun()) q.runs(); // => flush json and return fake results
    }
  }

  @TestFactory
  Stream<DynamicNode> tests_cpu() throws IOException {
    return testsFromFileTree(
      Paths.get(PelagoQueryTest.class.getResource("/tests").getPath()),
      (sql, resultFile) -> Assertions.assertDoesNotThrow(() -> {
        Repl.setCpuonly();
        System.out.println(sql);
        testQueryOrdered(sql, resultFile);
      })
    );
  }

  @TestFactory
  Stream<DynamicNode> tests_gpu() throws IOException {
    return testsFromFileTree(
      Paths.get(PelagoQueryTest.class.getResource("/tests").getPath()),
      (sql, resultFile) -> Assertions.assertDoesNotThrow(() -> {
        Repl.setGpuonly();
        testQueryOrdered(sql, resultFile);
      })
    );
  }

  @TestFactory
  Stream<DynamicNode> tests_hyb() throws IOException {
    return testsFromFileTree(
      Paths.get(PelagoQueryTest.class.getResource("/tests").getPath()),
      (sql, resultFile) -> Assertions.assertDoesNotThrow(() -> {
        Repl.setHybrid();
        testQueryOrdered(sql, resultFile);
      })
    );
  }

  /**
   * Helper function to create a list of tests based on a folder structure
   *
   * @param path path to a folder, root of the tree structure under consideration
   * @param test function to execute per sql query
   * @return a stream of DynamicTests
   *
   * @throws IOException for exception relating to traversing the root of the tree
   */
  private static Stream<DynamicNode> testsFromFileTree(Path path, BiConsumer<String, Path> test) throws IOException {
    return Stream.concat(
      Files.list(path)  // we want to control the order of traversal, otherwise we would use the Files.walk function
        .filter(Files::isRegularFile)
        .filter((x) -> x.getFileName().toString().endsWith(".sql"))
        .sorted()       // sorted in order to guarantee the order between different invocations
        .map((file) -> {
          try {
            // find file containing the verification set: same path/filename but extended with .resultset
            Path rFile = Paths.get(file.toString() + ".resultset");
            final Path resultFile = (Files.exists(rFile) && Files.isRegularFile(rFile)) ? rFile : null;
//            if (resultFile == null) return null;
            // clean the sql command from comments and final ';'
            String sql = new String(Files.readAllBytes(file));
            sql = sql.replaceAll("--[^\n]*", "").trim();
            assert(sql.lastIndexOf(';') == sql.length() - 1);
            final String q = sql.substring(0, sql.length() - 1);

            // create the test
            return (DynamicNode) dynamicTest(file.getFileName().toString(),
              () -> assertTimeoutPreemptively(
                  ofSeconds((PelagoTestConnectionFactory.isDebug) ? Long.MAX_VALUE/1000 : 2000),
                  () -> test.accept(q, resultFile)
              ));
          } catch (IOException e) {
            logger.warn(e.getMessage());
            return null;
          }
        })
        .filter(Objects::nonNull),
      Files.list(path)
        .filter((x) -> !x.getFileName().toString().equals("current"))
        .filter(Files::isDirectory)
        .sorted()       // sorted in order to guarantee the order between different invocations
        .map((file) -> {
          try {
            return (DynamicNode) dynamicContainer(file.getFileName().toString(), testsFromFileTree(file, test));
          } catch (IOException e) {
            logger.warn(e.getMessage());
            return null;
          }
        })
        .filter(Objects::nonNull));
  }
}
