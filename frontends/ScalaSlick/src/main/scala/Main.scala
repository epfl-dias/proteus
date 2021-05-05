/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

import demo.Tables

//object Tables extends {
//  // or just use object demo.Tables, which is hard-wired to the driver stated during generation
//  val profile = slick.jdbc.PostgresProfile
//} with demo.Tables
import demo.Tables.profile.api._

import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

object Main {
  //
  //  val db = Database.forURL("jdbc:avatica:remote:url=http://diascld43.iccluster.epfl.ch:8081;serialization=PROTOBUF",
  //    driver="org.apache.calcite.avatica.remote.Driver")

  def main(args: Array[String]): Unit = {
    //    db.createSession().

    //    Tables

    //    print(Tables.A)

    //    for {
    //      Tables.
    //    }
    //    def x(y: Rep[Int]) = for {
    //      c <- Tables.SsbmDate if c.dYear === y
    //    } yield (c.dDatekey).sum
    def x(y: Rep[Int]) = {for {
      c <- Tables.SsbmDate if c.dYear === y
    } yield (c.dDatekey)}.sum

    //
    //    val x2 = (for {
    //      e <- Tables.Employeesnum
    //      ec <- e.children
    //    } yield (1)).sum

    val url = "jdbc:avatica:remote:url=http://diascld41.iccluster.epfl.ch:8081;serialization=PROTOBUF"
    val db = Database.forURL(url, driver = "org.apache.calcite.avatica.remote.Driver")

    val y = Compiled(x _)
    println("asdasd--")
    val res = Await.result(db.run(y(1993).result), 60 seconds)
    println("asdasd")

    val res2 = Await.result(db.run(y(1994).result), 60 seconds)
    println("asdasd++")


    println("hello world")
    println(res2.head)


  }
}
