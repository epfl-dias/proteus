name := "ScalaSlick"

version := "0.1"

scalaVersion := "2.13.4"

val slickVersion = "3.3.3"

libraryDependencies ++= Seq(
  "com.typesafe.slick" %% "slick" % slickVersion,
  "com.typesafe.slick" %% "slick-codegen" % slickVersion,
  "org.apache.calcite.avatica" % "avatica" % "1.13.0",
  "org.slf4j" % "slf4j-api" % "1.7.13",
  "org.slf4j" % "slf4j-log4j12" % "1.7.13"
)

sourceGenerators in Compile += slick.taskValue // Automatic code generation on build

lazy val slick = taskKey[Seq[File]]("Generate Tables.scala")
slick := {
  val dir = (sourceManaged in Compile).value
  val outputDir = dir / "slick"
  val url = "jdbc:avatica:remote:url=http://diascld41.iccluster.epfl.ch:8081;serialization=PROTOBUF"
  val jdbcDriver = "org.apache.calcite.avatica.remote.Driver"
  val slickDriver = "slick.jdbc.MySQLProfile"
  val pkg = "demo"

  val cp = (dependencyClasspath in Compile).value
  val s = streams.value

  runner.value.run("slick.codegen.SourceCodeGenerator",
    cp.files,
    Array(slickDriver, jdbcDriver, url, outputDir.getPath, pkg),
    s.log).failed foreach (sys error _.getMessage)

  val file = outputDir / pkg / "Tables.scala"

  Seq(file)
}
