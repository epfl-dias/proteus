name := "SQLPlanner"

version := "0.1"

scalaVersion := "2.12.3"



// https://mvnrepository.com/artifact/org.json4s/json4s-jackson_2.10
val json4sJackson = "org.json4s" % "json4s-jackson_2.12" % "3.5.3"
// https://mvnrepository.com/artifact/org.apache.calcite/calcite-core
val calciteCore = "org.apache.calcite" % "calcite-core" % "1.14.0"
val calciteCSV = "org.apache.calcite" % "calcite-example-csv" % "1.14.0"
val slf4j = "org.slf4j" % "slf4j-api" % "1.7.13"
val slf4j_simple = "org.slf4j" % "slf4j-simple" % "1.7.13"

libraryDependencies += calciteCore
libraryDependencies += calciteCSV
libraryDependencies += slf4j
libraryDependencies += slf4j_simple
libraryDependencies += json4sJackson