package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.test.CalciteAssert;

import ch.epfl.dias.repl.Repl;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicNode;
import org.junit.jupiter.api.TestFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
  static final boolean isDebug = ManagementFactory.getRuntimeMXBean().getInputArguments().toString().indexOf("jdwp")>=0;

  @BeforeAll
  public static void init() throws SQLException {
    // warm-up connection and load classes
    PelagoTestConnectionFactory.get();
  }

  private CalciteAssert.AssertQuery testParseAndExecute(String sql) throws SQLException {
    return CalciteAssert.that()
      .with(PelagoTestConnectionFactory.get())
      .query(sql)
      .explainContains("PLAN=PelagoToEnumerableConverter")
      .runs()
      ;
  }

  public void testQueryOrdered(String sql, Path resultFile) throws SQLException, IOException {
    if (resultFile != null && !Repl.isMockRun()){
      List<String> lines = Files.readAllLines(resultFile);
      testParseAndExecute(sql)
        .returnsOrdered(lines.toArray(new String[lines.size()]));
    } else {
      if (resultFile != null) logger.debug("Result set not verified (reason: running in mock mode)");
      testParseAndExecute(sql);
    }
  }

  @TestFactory
  Stream<DynamicNode> tests() throws IOException {
    return testsFromFileTree(
        Paths.get(PelagoQueryTest.class.getResource("/tests").getPath()),
        (sql, resultFile) -> Assertions.assertDoesNotThrow(() -> testQueryOrdered(sql, resultFile))
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
        .filter((x) -> !(x.getFileName().endsWith(".resultset")))
        .sorted()       // sorted in order to guarantee the order between different invocations
        .map((file) -> {
          try {
            // find file containing the verification set: same path/filename but extended with .resultset
            Path rFile = Paths.get(file.toString() + ".resultset");
            final Path resultFile = (Files.exists(rFile) && Files.isRegularFile(rFile)) ? rFile : null;

            // clean the sql command from comments and final ';'
            String sql = new String(Files.readAllBytes(file)).trim();
            sql = sql.replaceAll("--[^\n]*", "");
            assert(sql.lastIndexOf(';') == sql.length() - 1);
            final String q = sql.substring(0, sql.length() - 1);

            // create the test
            return (DynamicNode) dynamicTest(file.getFileName().toString(),
              () -> assertTimeoutPreemptively(
                  ofSeconds((isDebug) ? Long.MAX_VALUE/1000 : 200),
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
