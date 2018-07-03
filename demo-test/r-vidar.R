library(ViDaR)
library(DBI)
library(dbplyr)
library(dplyr)
library(ggplot2)
library(rlang)
library(jsonlite)

### DBI ###
# establishing the connection
driverClass <- "org.apache.calcite.avatica.remote.Driver"
driverLocation <- "src/avatica-1.11.0.jar"
connectionString <- "jdbc:avatica:remote:url=http://localhost:8081;serialization=PROTOBUF"

con <- dbConnect(ViDaR(driverClass = driverClass, driverLocation = driverLocation))

test_noheader <- readcsv(connection = con, path = "demo-test/test.csv", linehint = 5, header = FALSE)

test_header <- readcsv(connection = con, name="test_header", path = "demo-test/test_header.csv", linehint = 5)

test_fields <- readcsv(connection = con, name="test_fields", path = "/random_path/test_header.csv", linehint = 5, colClasses = c("integer", "varchar"))

# TODO - unparsing nested tables
#json <- '{"employees": { "path": "inputs/json/employees-flat.json", "type": { "type": "bag", "inner": { "type": "record", "attributes": [{ "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json", "attrName": "name", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json", "attrName": "age", "attrNo": 2 }, { "type": { "type": "list", "inner": { "type": "record", "attributes": [ { "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json", "attrName": "name2", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json", "attrName": "age2", "attrNo": 2 } ] } }, "relName": "inputs/json/employees-flat.json", "attrName": "children", "attrNo": 3 }] } }, "plugin": { "type": "json", "lines": 3, "policy": 2 } } }'
#test_json <- readjson(connection = con, name = "test_json", json = json)

### dplyr ###
# creating placeholder tables (query results in 0 rows fetched (WHERE 0=1))
dates <- tbl(con, "ssbm_date")
lineorder <- tbl(con, "ssbm_lineorder")
customer <- tbl(con, "ssbm_customer")
supplier <- tbl(con, "ssbm_supplier")
part <- tbl(con, "ssbm_part")

#REPLACE THE TABLES FROM SCHEMA WITH TABLES CREATED IN R
dates_sch <- paste0("d_datekey:int,d_date:string,d_dayofweek:string,d_month:string,",
                    "d_year:int,d_yearmonthnum:int,d_yearmonth:string,d_daynuminweek:int,",
                    "d_daynuminmonth:int,d_daynuminyear:int,d_monthnuminyear:int,d_weeknuminyear:int,",
                    "d_sellingseason:string,d_lastdayinweekfl:boolean,d_lastdayinmonthfl:boolean,",
                    "d_holidayfl:boolean,d_weekdayfl:boolean")

dates <- readcsv(connection = con, fields = dates_sch, path = "/path/to/dates.csv", linehint = 5000, name = "date")

# dates_csv <- tbl(con, "dates_csv")
# lineorder_csv <- tbl(con, "lineorder_csv")
# customer_csv <- tbl(con, "customer_csv")
# supplier_csv <- tbl(con, "supplier_csv")
# part_csv <- tbl(con, "part_csv")

dates %>% filter(d_yearmonthnum==199401L) %>% select(d_yearmonthnum, d_datekey)

dates %>% filter(d_yearmonthnum==199401L) %>% select(d_yearmonthnum, d_datekey) %>% count()

dates %>% summarise(sum1=sum(d_year, na.rm = TRUE))

dates %>% inner_join(lineorder, by = c("d_datekey"="lo_orderdate")) %>% count()

supplier %>% filter(s_name == 'Supplier#000047861')

# selected supplier profit over years (monthly granularity) - line chart
supplier_profit <- inner_join(supplier, lineorder, by=c("s_suppkey"="lo_suppkey")) %>%
  filter(s_name=='Supplier#000047861') %>%
  inner_join(., dates, by=c("lo_orderdate"="d_datekey"))%>%
  inner_join(., part, by=c("lo_partkey"="p_partkey")) %>%
  group_by(d_yearmonthnum) %>%
  summarize(profit=sum(lo_revenue-lo_supplycost, na.rm = TRUE))

supplier_profit %>% collect() %>% ggplot(., aes(x=d_yearmonthnum, y=profit, group=1)) +
  geom_line() +
  ggtitle(paste("Profits for Supplier#000047861"))


# profit for selected product in regions in 1997 - barchart - execution bug
# product_year_profit <- inner_join(dates, lineorder, by=c("d_datekey"="lo_orderdate")) %>%
#   inner_join(., supplier, by=c("lo_suppkey"="s_suppkey")) %>%
#   inner_join(., customer, by=c("lo_custkey"="c_custkey")) %>%
#   inner_join(., part, by=c("lo_partkey"="p_partkey")) %>%
#   filter(c_region=='AMERICA' & s_region=='AMERICA' & p_mfgr=='MFGR#1' & d_year==1997L) %>%
#   group_by(d_yearmonthnum) %>%
#   summarize(profit=sum(lo_revenue-lo_supplycost, na.rm = TRUE))
#
# product_year_profit %>% collect() %>% ggplot(.) + geom_bar(aes(x = d_yearmonthnum, y = profit), stat="identity") +
#   ggtitle("Profits for MFGR#1 in USA")


# histogram - suppliers by profit in 1997 and 1998
suppliers_by_profit <- inner_join(lineorder, supplier, by=c("lo_suppkey"="s_suppkey")) %>%
  filter(s_city=='UNITED KI1') %>%
  inner_join(., dates, by=c("lo_orderdate"="d_datekey")) %>%
  filter(d_year>=1996L) %>%
  inner_join(., part, by=c("lo_partkey"="p_partkey")) %>%
  group_by(d_year, s_name) %>%
  summarize(profit=sum((lo_revenue-lo_supplycost), na.rm = TRUE))

suppliers_by_profit %>% collect() %>% ggplot(., aes(x=profit, group=d_year, fill=d_year)) +
  geom_histogram(alpha=0.5, position="identity") +
  ggtitle("Distribution of supplier profits in UNITED KI1, after 1996")



# pure SQL query - DBI
query <- paste0("SELECT sum(d_year), count(*), sum(lo_revenue-lo_supplycost) as profit ",
                "from dates, customer, supplier, part, lineorder ",
                "where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_partkey=p_partkey ",
                "and lo_orderdate=d_datekey and c_region='AMERICA' and s_region='AMERICA' ",
                "and (p_mfgr='MFGR#1' or p_mfgr='MFGR#2')")

tmp_tbl <- dbGetQuery(con, query) # wraps around dbSendQuery and dbFetch
tmp_tbl

# equivalent to previous SQL query
result <- inner_join(lineorder, customer, by=c("lo_custkey"="c_custkey")) %>%
  inner_join(., supplier, by=c("lo_suppkey"="s_suppkey")) %>%
  inner_join(., part, c("lo_partkey"="p_partkey")) %>%
  inner_join(., dates, by=c("lo_orderdate"="d_datekey")) %>%
  filter(c_region=='AMERICA' & s_region=='AMERICA' & (p_mfgr=='MFGR#1' | p_mfgr=='MFGR#2')) %>%
  summarize(sumY=sum(d_year, na.rm = TRUE), cnt=count(), profit=sum(lo_revenue-lo_supplycost, na.rm = TRUE))

result



# listing the tables in available schemas
#dbListTables(con)

#dbExistsTable(con, "dates")
#dbExistsTable(con, "dates1")
#dbListFields(con, "dates")

dbDisconnect(con)

path <- "/home/sanca/ViDa/pelago/opt/raw/inputs/ssbm100/"

start<-Sys.time()
lineorder <- fread(paste0(path,"lineorder2.tbl"), sep = '|', header = FALSE)
dates <- fread(paste0(path,"date2.tbl"), sep = '|', header = FALSE)
customer <- fread(paste0(path,"customer2.tbl"), sep = '|', header = FALSE)
part <- fread(paste0(path,"part2.tbl"), sep = '|', header = FALSE)
supplier <- fread(paste0(path,"supplier2.tbl"), sep = '|', header = FALSE)
end<-Sys.time()

names(lineorder) <- names(schema2tbl("lineorder"))
names(dates) <- names(schema2tbl("dates"))
names(customer) <- names(schema2tbl("customer"))
names(part) <- names(schema2tbl("part"))
names(supplier) <- names(schema2tbl("supplier"))

start<-Sys.time()
hist_data = inner_join(lineorder, customer, by=c("lo_custkey"="c_custkey")) %>%
  inner_join(., supplier, by=c("lo_suppkey"="s_suppkey")) %>%
  inner_join(., part, c("lo_partkey"="p_partkey")) %>%
  inner_join(., dates, by=c("lo_orderdate"="d_datekey")) %>%
  filter(c_region=='AMERICA' & s_region=='AMERICA' & p_mfgr=='MFGR#1' & d_date=='January 6, 1994') %>%
  select(lo_revenue)
end<-Sys.time()


