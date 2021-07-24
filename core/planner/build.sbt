name := "planner"

version := "0.1"

scalaVersion := "2.12.3"

val calciteVersion = "1.27.0"

// https://mvnrepository.com/artifact/org.apache.calcite/calcite-core
// Include Calcite Core
libraryDependencies += "org.apache.calcite" % "calcite-core" % calciteVersion
// Also include the tests.jar of Calcite Core as a dependency to our testing jar
libraryDependencies += "org.apache.calcite" % "calcite-core" % calciteVersion % Test classifier "tests"

// Calcite DDL Parser
// https://mvnrepository.com/artifact/org.apache.calcite/calcite-server
libraryDependencies += "org.apache.calcite" % "calcite-server" % calciteVersion

// https://mvnrepository.com/artifact/org.json4s/json4s-jackson_2.10
libraryDependencies += "org.json4s" % "json4s-jackson_2.12" % "3.5.3"

// slf4j
libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.13"
libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "1.7.13"

// https://mvnrepository.com/artifact/org.apache.calcite.avatica/avatica
libraryDependencies += "org.apache.calcite.avatica" % "avatica-server" % "1.17.0"
// https://mvnrepository.com/artifact/com.google.guava/guava
libraryDependencies += "com.google.guava" % "guava" % "19.0"
// https://mvnrepository.com/artifact/au.com.bytecode/opencsv
libraryDependencies += "au.com.bytecode" % "opencsv" % "2.4"
// https://mvnrepository.com/artifact/commons-io/commons-io
libraryDependencies += "commons-io" % "commons-io" % "2.4"
// https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.9.4"

//// https://mvnrepository.com/artifact/junit/junit
//libraryDependencies += "junit" % "junit" % "4.12" % Test
//libraryDependencies += "junit" % "junit" % "4.12" % Test

// https://mvnrepository.com/artifact/org.junit.jupiter/junit-jupiter-api
libraryDependencies += "org.junit.jupiter" % "junit-jupiter-api" % "5.3.1" % Test
libraryDependencies += "org.junit.jupiter" % "junit-jupiter-params" % "5.3.1" % Test

// junit tests (invoke with `sbt test`)
libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test"

fork in Test := true

// disable tests during `sbt assembly`
test in assembly := {}

assembly / assemblyJarName := "proteusplanner.jar"

//fork in Test := true
//baseDirectory in Test := file("/cloud_store/periklis/pelago_cidr2/opt/raw")

// Credits: https://github.com/sbt/sbt/issues/1789#issue-53027223
def listFilesRecursively(dir: File): Seq[File] = {
  val list = sbt.IO.listFiles(dir)
  list.filter(_.isFile) ++ list
    .filter(_.isDirectory)
    .flatMap(listFilesRecursively)
}

sourceGenerators in Compile += Def.task {
  val codegenDir = baseDirectory.value / "src" / "main" / "codegen"
  // * Create a cached function which generates the output files
  //   only if the input files have changed.
  // * The first parameter is a file instance of the path to
  //   the cache folder
  // * The second parameter is the function to process the input
  //   files and return the output files
  val cached = FileFunction.cached(
    baseDirectory.value / ".cache" / "codegen"
  ) { (in: Set[File]) =>
    val fmppFolder = (sourceManaged in Compile).value / "fmpp"
    val javaccFolder = (sourceManaged in Compile).value / "javacc"
    val externalFmppFolder = (sourceManaged in Compile).value / "external-fmpp"
    (dependencyClasspath in Compile).value filter {
      _.data.getName.startsWith("calcite-core")
    } foreach { f =>
      IO.unzip(f.data, externalFmppFolder)
    }
    //  Def.task {
    val templatesFile = externalFmppFolder / "codegen" / "templates"

    if (
      fmpp.tools.CommandLine.execute(
        Array(
          "-C",
          (codegenDir / "config.fmpp").toString,
          "-S",
          templatesFile.toString,
          "-O",
          fmppFolder.toString,
          "--data",
          "tdd(" + (codegenDir / "config.fmpp").toString + "), default: tdd(" + templatesFile / ".." / "default_config.fmpp" + ")"
        ),
        null,
        null
      ) != 0
    ) {
      throw new Exception("fmpp failed");
    }
    if (
      org.javacc.parser.Main.mainProgram(
        Array(
          "-STATIC=false",
          "-LOOKAHEAD=2",
          "-OUTPUT_DIRECTORY=" + javaccFolder.toString,
          (fmppFolder / "javacc" / "Parser.jj").toString
        )
      ) != 0
    ) {
      throw new Exception("javacc failed");
    }
    listFilesRecursively(javaccFolder).toSet
  }
  cached(
    listFilesRecursively(codegenDir).toSet + baseDirectory.value / "build.sbt"
  ).toSeq
}.taskValue

resolvers += Resolver.jcenterRepo
testOptions += Tests.Argument(jupiterTestFramework, "-q")

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.first
  case PathList("META-INF", xs @ _*)             => MergeStrategy.discard
  case PathList("org", "apache", "calcite", "sql", "ddl", xs @ _*) =>
    MergeStrategy.first
  case x if x.endsWith("module-info.class") => MergeStrategy.discard
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
