name := "clotho"

version := "0.1"

scalaVersion := "2.12.3"

val calciteVersion = "1.16.0"

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
libraryDependencies += "org.apache.calcite.avatica" % "avatica-server" % "1.11.0"
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

//fork in Test := true
//baseDirectory in Test := file("/cloud_store/periklis/pelago_cidr2/opt/raw")

resolvers += Resolver.jcenterRepo
testOptions += Tests.Argument(TestFrameworks.JUnit, "-q")

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.first
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  // FIXME: when issue #4 is resolved, replace with commented portion
  case x => MergeStrategy.first
  // case x =>
  //   val oldStrategy = (assemblyMergeStrategy in assembly).value
  //   oldStrategy(x)
}

