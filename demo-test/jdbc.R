library(RJDBC)
library(dplyr)
library(DBI)

driverClass <- "ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver"
driverLocation <- "/home/sanca/sqlline/sqlline/bin/SQLPlanner-assembly-0.1.jar"
connectionString <- "jdbc:pelago:model=/home/sanca/sqlline/sqlline/bin/schema.json"


drv <- JDBC(driverClass, driverLocation)
conn <- dbConnect(drv, connectionString)


statement <- "!tables"
s <- .jcall(conn@jc, "Ljava/sql/Statement;", "createStatement")
r <- .jcall(s, "Ljava/sql/ResultSet;", "executeQuery", check=FALSE)


res <- dbCreateTable(conn, "test5432", c(b="integer"))

# === VIDAR JDBC === #


con <- dbConnect(ViDaR())
emp_tbl <- tbl(con, "employees")


schema2tbl("employees", con)


dbListTables(con)
dbCreateTable(con, "iris", iris)


dbListFields(con, "employees")

copy_to(con, iris, "iris")

dbSendQuery(con, "SELECT * FROM `emp` WHERE (`age` > 18.0)")
