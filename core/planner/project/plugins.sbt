resolvers += Resolver.jcenterRepo
addSbtPlugin("net.aichler" % "sbt-jupiter-interface" % "0.7.0")

libraryDependencies += "net.sourceforge.fmpp" % "fmpp" % "0.9.16"
// https://mvnrepository.com/artifact/net.java.dev.javacc/javacc
libraryDependencies += "net.java.dev.javacc" % "javacc" % "7.0.4"
