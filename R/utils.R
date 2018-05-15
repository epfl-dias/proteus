# DBI Utils

# Utility function for preprocessing the text of the query -
# escaped quotes are deletd from the query
textProcessQuery <- function(query) {
  ret_query <- gsub("\"","", query)
  ret_query <- gsub("\\\n", "", ret_query)
  #ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)

  return(ret_query)
}

# Util function for extracting the table name in FROM clause
extractFrom <- function(query) {
  from <- strsplit(textProcessQuery(query), "FROM ")[[1]][2]
  from <- strsplit(from, " ")[[1]][1]
  return(from)
}

# TO BE DEPRECATED AND REPLACED WITH CALCITE HANDLER
# Table name - find path of the schema
getPath <- function(table_name){
  path <- "/home/sanca/ViDa/pelago/src/SQLPlanner/src/main/resources/"

  json<-jsonlite::read_json(paste0(path,"schema.json"))

  for(schema in json$schemas){
    for(tbl in list.files(paste0(path,schema$operand$directory)))
      if(strsplit(tbl,"\\.")[[1]][1]==table_name){
        return(paste0(path,schema$operand$directory,'/',table_name,'.csv'))
      }
  }

  return(NULL)
}

# mapping between types in CSV and R types
type_map <- list(int="integer(0)", string="character(0)", boolean="logical(0)")

# for case of creating a tbl (return 0 rows), R magic with lazy evaluation
schema2tbl <- function(table){

  # TEST PURPOSES FOR NESTED SCHEMAS
  if(table=="emp") {
    emp_jsn = '{"name":"string", "age":"int", "children":[{"name2":"string", "age2":"int"}]}'
    df_emp <- data.frame(jsonlite::fromJSON(emp_jsn, flatten = TRUE, simplifyDataFrame = TRUE))
    emp <- as.tbl(df_emp)
    return(emp)
  }

  suppressWarnings(tmp <- read.csv(getPath(table)))

  build_cmd <- "list("

  for(col in colnames(tmp)){

    name <- strsplit(col,"\\.")[[1]][1]
    type <- strsplit(col,"\\.")[[1]][2]

    build_cmd <- paste0(build_cmd, name, "=", type_map[type], ",")
  }

  build_cmd<-substr(build_cmd, 1, nchar(build_cmd)-1)
  build_cmd<-paste0(build_cmd,")")

  return(as.tbl(data.frame(lazyeval::lazy_eval(build_cmd))))
}

#' @export
#' @rdname tbl_lazy
simulate_vidar <- function() {
  structure(
    list(),
    class = c("ViDaRConnection", "DBIConnection")
  )
}
