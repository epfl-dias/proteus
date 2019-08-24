package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.test.CalciteAssert;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.repl.Repl;
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
    ResultSet rs = PelagoTestConnectionFactory.get()
      .createConnection()
      .createStatement()
      .executeQuery("SELECT * FROM SessionTimings");

    ResultSetMetaData rsmd = rs.getMetaData();
    int columnsN = rsmd.getColumnCount();
    while (rs.next()){
      // Keep lines in a string and print whole lines each time,
      // otherwise in junit prints each part a separate info line.
      ImmutableList.Builder builder = ImmutableList.<String>builder();
      for (int i = 1 ; i <= columnsN ; ++i){
        String columnValue = rs.getString(i);
        builder.add(rsmd.getColumnName(i) + "=" + columnValue);
      }
      System.out.println(String.join("; ", builder.build()));
    }
  }

  private CalciteAssert.AssertQuery testParseAndExecute(String sql) throws SQLException {
    return CalciteAssert.that()
      .with(PelagoTestConnectionFactory.get())
      .query(sql)
      .explainContains("PLAN=PelagoToEnumerableConverter")
      ;
  }

  public static String getModeAsString(){
    if (Repl.isHybrid()) return "hyb";
    if (Repl.isGpuonly()) return "gpu";
    if (Repl.isCpuonly()) return "cpu";
    assert(false);
    return "";
  }

  public void testQueryOrdered(String sql, Path resultFile) throws SQLException, IOException {
    CalciteAssert.AssertQuery q = testParseAndExecute(sql);

    String explainContains = "PLAN=PelagoToEnumerableConverter";
    if (resultFile != null) {
      final String suffix = ".resultset";
      String filename = resultFile.toString();
      assert(filename.endsWith(suffix));
      filename = filename.substring(0, filename.length() - suffix.length());
      final Path p = Paths.get(filename + "." + getModeAsString() + ".plan");
      if (Files.exists(p) && Files.isRegularFile(p)){
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
    }
    assert(explainContains.startsWith("PLAN=PelagoToEnumerableConverter"));
    q = q.explainMatches("EXCLUDING ATTRIBUTES ", CalciteAssert.checkResultContains(explainContains));

    if (resultFile != null && !Repl.isMockRun()){
      List<String> lines = Files.readAllLines(resultFile);
      q.returnsOrdered(lines.toArray(new String[lines.size()]));
    } else {
      if (resultFile != null) logger.debug("Result set not verified (reason: running in mock mode)");
      q.runs(); // => flush json and return fake results
    }
  }

  @TestFactory
  Stream<DynamicNode> tests_cpu() throws IOException {
    return testsFromFileTree(
      Paths.get(PelagoQueryTest.class.getResource("/tests").getPath()),
      (sql, resultFile) -> Assertions.assertDoesNotThrow(() -> {
        Repl.setCpuonly();
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
        .filter((x) -> x != null),
      Files.list(path)
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
        .filter((x) -> x != null));
  }
}
