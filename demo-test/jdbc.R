library(RJDBC)
library(dplyr)

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
date_tbl <- tbl(con, "ssbm_date")


r <- dbSendQuery(con, "SELECT * FROM ssbm_date")


dbListTables(con)
dbCreateTable(con, "iris", iris)


dbListFields(con, "ssbm_date")

copy_to(con, iris, "iris")

dbSendQuery(con, "SELECT * FROM `emp` WHERE (`age` > 18.0)")
