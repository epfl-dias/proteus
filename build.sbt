name := "SQLPlanner"

version := "0.1"

scalaVersion := "2.12.3"

val calciteVersion = "1.16.0"

// https://mvnrepository.com/artifact/org.json4s/json4s-jackson_2.10
val json4sJackson = "org.json4s" % "json4s-jackson_2.12" % "3.5.3"
// https://mvnrepository.com/artifact/org.apache.calcite/calcite-core
val calciteCore = "org.apache.calcite" % "calcite-core" % calciteVersion
// https://mvnrepository.com/artifact/org.apache.calcite/calcite-server
val calciteServer = "org.apache.calcite" % "calcite-server" % calciteVersion
//val calciteCSV = "org.apache.calcite" % "calcite-example-csv" % "1.14.0"
val slf4j = "org.slf4j" % "slf4j-api" % "1.7.13"
val slf4j_simple = "org.slf4j" % "slf4j-simple" % "1.7.13"

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
libraryDependencies += calciteCore
libraryDependencies += calciteServer
//libraryDependencies += calciteCSV
libraryDependencies += slf4j
libraryDependencies += slf4j_simple
libraryDependencies += json4sJackson

// https://mvnrepository.com/artifact/junit/junit
libraryDependencies += "junit" % "junit" % "4.12" % Test

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.first
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}