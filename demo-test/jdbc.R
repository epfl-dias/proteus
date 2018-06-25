library(RJDBC)
library(dplyr)
library(DBI)
library(ViDaR)

driverClass <- "org.apache.calcite.avatica.remote.Driver"
driverLocation <- "src/avatica-1.11.0.jar"
connectionString <- "jdbc:avatica:remote:url=http://localhost:8081;serialization=PROTOBUF"

# drv <- JDBC(driverClass, driverLocation, identifier.quote = '`')
# conn <- dbConnect(drv, connectionString)
#
# statement <- "!tables"
# s <- .jcall(conn@jc, "Ljava/sql/Statement;", "createStatement")
# r <- .jcall(s, "Ljava/sql/ResultSet;", "executeQuery", check=FALSE)
#
# res <- dbCreateTable(conn, "test5432", c(b="integer"))

# === VIDAR JDBC === #


con <- dbConnect(ViDaR(driverClass = driverClass, driverLocation = driverLocation))

#emp_tbl <- tbl(con, "employees")


#dbSendUpdate(con, "create table Test1234(a integer, b integer) jplugin `{\'plugin\':{ \'type\':\'block\', \'linehint\':200000 }, \'file\':\'/inputs/csv.csv\'}`")

#dbSendQuery(con, "select * from Test1234")

#test1234 <- tbl(con, "Test1234")
test1234 %>% filter(a > 0.0) %>% summarise(colb = b)

dates <- tbl(con, "ssbm_date")

#schema2tbl("employees", con)
test <- list(a="integer", b="varchar")
dbCreateTable(conn = con, name = "Test", fields = test, path = "mock.csv", linehint = 20000, type = "csv")

dbListTables(con)
dbCreateTable(con, "iris3", iris, path = "mock.csv", linehint = 20000, type = "csv")

iris <- tbl(con, "iris")
iris %>% filter(Sepal.Length > 5.0)

dbListFields(con, "employees")

copy_to(con, iris, "iris")

csv <- readcsv(connection = con, path = "mock.csv", fields = list(a="integer", b="varchar"), linehint = 20000, local = TRUE)

