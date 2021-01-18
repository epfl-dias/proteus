#!/usr/bin/env Rscript
library(ViDaR)
library(DBI)
library(dbplyr)
library(dplyr)
library(ggplot2)
library(rlang)
library(optparse)

option_list = list(
  make_option(c("-d", "--driverClass"), type="character", default="org.apache.calcite.avatica.remote.Driver", 
              help="jdbc driver", metavar="character"),
  make_option(c("-j", "--driverJar"), type="character", default="opt/lib/avatica-1.13.0.jar", 
              help="jdbc driver jar", metavar="character"),
  make_option(c("-s", "--server"), type="character", default="localhost", 
              help="server url", metavar="character"),
  make_option(c("-p", "--port"), type="character", default="8081", 
              help="server port", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# connection parameters
driverClass <- opt$driverClass
driverLocation <- opt$driverJar
connectionString <- paste("jdbc:avatica:remote:url=http://", opt$server, ":", opt$port, ";serialization=PROTOBUF", sep="")

# establishing the connection
con <- dbConnect(ViDaR(driverClass = driverClass, driverLocation = driverLocation), connectionString = connectionString)

vidar.default.connection(con)

# creating table only from csv, linehint still necessary
test_noheader <- read.csv("../../src/frontends/R/demo-test/test.csv", nrows = 3, policy = 3, header = FALSE, sep = ',', conn = con)
test_noheader

test_header <- read.csv("../../src/frontends/R/demo-test/test_header.csv", nrows = 3, policy = 3, sep=',', quote="'", conn = con)
test_header

# creating from specified column classes
test_fields <- read.csv("../../src/frontends/R/demo-test/test_header.csv", nrows = 3, policy = 3, colClasses = c("integer", "integer", "character", "logical"), quote="'", conn = con)
test_fields

# creating if there exists an external file specification
col.names <- c("d_datekey", "d_date", "d_dayofweek", "d_month", "d_year", 
  "d_yearmonthnum", "d_yearmonth", "d_daynuminweek", "d_daynuminmonth", 
  "d_daynuminyear", "d_monthnuminyear", "d_weeknuminyear", "d_sellingseason", 
  "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl")

colClasses <- c("integer", "character", "character", "character", "integer",
  "integer", "character", "integer", "integer",
  "integer", "integer", "integer", "character",
  "integer", "integer", "integer", "integer")
  # "logical", "logical", "logical", "logical")

test_sch <- read.csv("inputs/ssbm100/raw/date2.tbl", col.names = col.names, colClasses = colClasses, nrows = 2556, policy = 10, sep='|', header = FALSE, quote="", conn = con)
test_sch

# creating table from json type specification, this needs to be wrapped with something more convenient
json_type <- paste0('{"type": "bag", "inner": { "type": "record", "attributes":',
               ' [{ "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json",',
               ' "attrName": "name", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json",',
               ' "attrName": "age", "attrNo": 2 }, { "type": { "type": "list", "inner": { "type": "record",',
               ' "attributes": [ { "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json",',
               ' "attrName": "name2", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json",',
               ' "attrName": "age2", "attrNo": 2 } ] } }, "relName": "inputs/json/employees-flat.json",',
               ' "attrName": "children", "attrNo": 3 }] } }')

test_json <- fromJSON("inputs/json/employees-flat.json", ndjson=TRUE, fields=json_type, conn = con)
test_json %>% select (name, age)

q()

# creating placeholder tables
dates <- tbl(con, "ssbm_date")
lineorder <- tbl(con, "ssbm_lineorder")
supplier <- tbl(con, "ssbm_supplier")

# loading from "csv", in this case the difference is that the csv has trailing separator, requiring additional column (marked as "nl"- newline)
customer <- read.csv("inputs/ssbm100/raw/customer.tbl", name="ssbm_customer1", nrows = 3000000, policy = 4,
                     colNames = c("c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment", "nl"), header = FALSE, sep = '|', quote="", conn = con)

part <- read.csv("inputs/ssbm100/raw/part.tbl", name="ssbm_part1", nrows = 1400000, policy = 4
                 colNames = c("p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container", "nl"), header = FALSE, sep = '|', brackets=FALSE, quote="", conn = con)

### read.csv (default R implementation) - reminder, this takes quite a long time to execute ###
# customer_csv <- read.csv("inputs/ssbm100/raw/customer.tbl", header = FALSE, sep = '|', col.names = c("c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment", "nl"), quote="")
# part_csv <- read.csv("inputs/ssbm100/raw/part.tbl", header = FALSE, sep = '|', col.names = c("p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container", "nl"))

# customer_csv %>% filter(c_nation=='MOROCCO') %>% select(c_name, c_phone)
customer %>% filter(c_nation=='MOROCCO') %>% select(c_name, c_phone)


# loading the nested table as well, no support yet for nested queries
employees <- tbl(con, "employees")
employees %>% select(age)


# dplyr demonstration - filtering, summarising, selecting, joining
dates %>% filter(d_yearmonthnum==199401L) %>% select(d_yearmonthnum, d_datekey)

dates %>% filter(d_yearmonthnum==199401L) %>% select(d_yearmonthnum, d_datekey) %>% count()

dates %>% summarise(sum1=sum(d_year, na.rm = TRUE))

dates %>% inner_join(lineorder, by = c("d_datekey"="lo_orderdate")) %>% count()

supplier %>% filter(s_name == 'Supplier#000047861')

# some more advanced demonstration
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

# query lasts for too long
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



# SQL vs DBI+dplyr
# SQL
query <- paste0("SELECT sum(d_year), count(*), sum(lo_revenue-lo_supplycost) as profit ",
                "from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part, ssbm_lineorder ",
                "where lo_custkey=c_custkey and lo_suppkey=s_suppkey and lo_partkey=p_partkey ",
                "and lo_orderdate=d_datekey and c_region='AMERICA' and s_region='AMERICA' ",
                "and (p_mfgr='MFGR#1' or p_mfgr='MFGR#2')")

tmp_tbl <- dbGetQuery(con, query)
tmp_tbl

# equivalent to previous SQL query
result <- inner_join(lineorder, customer, by=c("lo_custkey"="c_custkey")) %>%
  inner_join(., supplier, by=c("lo_suppkey"="s_suppkey")) %>%
  inner_join(., part, c("lo_partkey"="p_partkey")) %>%
  inner_join(., dates, by=c("lo_orderdate"="d_datekey")) %>%
  filter(c_region=='AMERICA' & s_region=='AMERICA' & (p_mfgr=='MFGR#1' | p_mfgr=='MFGR#2')) %>%
  summarize(sumY=sum(d_year, na.rm = TRUE), cnt=n(), profit=sum(lo_revenue-lo_supplycost, na.rm = TRUE))

result

dbDisconnect(con)


