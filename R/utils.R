# DBI Utils

# Utility function for preprocessing the text of the query -
# escaped quotes are deletd from the query
# WARNING: MAKE SURE THE QUERY IS PREPROCESSED ONLY ONCE!
textProcessQuery <- function(query, quoteChar = "") {
  ret_query <- query
  ret_query <- gsub("\"", quoteChar, ret_query)
  ret_query <- gsub("%", "\\\"", ret_query)
  ret_query <- gsub("\\\n", " ", ret_query)
  ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)
  #ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)
  ret_query <- gsub("LIMIT 0", "", ret_query)

  #TODO: temporarly disable
  return(ret_query)
  #return(query)
}

# Generating random table name
random_table_name <- function(n = 10) {
  paste0(sample(letters, n, replace = TRUE), collapse = "")
}

# Util function for extracting the table name in FROM clause
extractFrom <- function(query, quoteChar = "") {
  from <- strsplit(textProcessQuery(query, quoteChar), "FROM ")[[1]][2]
  from <- strsplit(from, " ")[[1]][1]
  from <- gsub("`", "", from)
  return(from)
}

# for case of creating a tbl (return 0 rows), R magic with lazy evaluation
# WILL NEED TO COVER THE CASE FOR NON-STANDARD TYPES (e.g. generate dataFrame first, then flatten it if nested types are present)
schema2tbl <- function(table, con, debug = FALSE){

  # TEST PURPOSES FOR NESTED SCHEMAS
  if(debug) {
    emp_jsn = '{"name":"string", "age":"int", "children":[{"name2":"string", "age2":"int"}]}'
    df_emp <- data.frame(jsonlite::fromJSON(emp_jsn, flatten = TRUE, simplifyDataFrame = TRUE))
    emp <- as.tbl(df_emp)
    return(emp)
  }

  metaData <- .jcall(con@jc, "Ljava/sql/DatabaseMetaData;", "getMetaData", check=FALSE)
  resultSet <- .jcall(metaData, "Ljava/sql/ResultSet;", "getColumns",
                      .jnull("java/lang/String"), .jnull("java/lang/String"), table, "%", check=FALSE)


  build_cmd <- c()

  colName <- character()
  colType <- character()
  while(.jcall(resultSet, "Z", "next")) {
    cn <- .jcall(resultSet, "S", "getString", "COLUMN_NAME")
    ct <- .jcall(resultSet, "S", "getString", "TYPE_NAME")

    if(!grepl("^recordtype", tolower(ct)))
      build_cmd <- c(build_cmd, paste0(cn, "=", mapJDBCType(ct)))
    else
      build_cmd <- c(build_cmd, parseRecordTypeList(ct, cn, con, table))

    colName <- c(colName, cn)
    colType <- c(colType, ct)
  }

  cmd <- paste0("data.frame(", paste(build_cmd, collapse = ","), ")")

  on.exit(.jcall(resultSet, "V", "close"))

  return(as.tbl(lazyeval::lazy_eval(cmd)))
}

# TODO - future work for creating JSON, R->Pelago
# json <- paste0('{"employees": { "path": "inputs/json/employees-flat.json",',
#               # ' "type": { "type": "bag", "inner": { "type": "record", "attributes":',
#                ' [{ "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json",',
#                ' "attrName": "name", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json",',
#                ' "attrName": "age", "attrNo": 2 }, { "type": { "type": "list", "inner": { "type": "record",',
#                ' "attributes": [ { "type": { "type": "string" }, "relName": "inputs/json/employees-flat.json",',
#                ' "attrName": "name2", "attrNo": 1 }, { "type": { "type": "int" }, "relName": "inputs/json/employees-flat.json",',
#                ' "attrName": "age2", "attrNo": 2 } ] } }, "relName": "inputs/json/employees-flat.json",',
#                ' "attrName": "children", "attrNo": 3 }] } }, "plugin": { "type": "json", "lines": 3, "policy": 2 } } }')
#
# quote_str <- function(string, quote_char = "\"") {
#   return(paste0(quote_char, string, quote_char))
# }
#
# list2json <- function(fields, name, path, linehint, quo = "\"", type = "bag") {
#   output <- "{"
#
#   output <- paste0(output, quote_str(name), ": { ", quote_str("path"), ": ", quote_str(path), ", ", quote_str("type"), ": {", quote_str("type"),": ")
#   output <- paste0(output, quote_str(type), ", ", quote_str("inner"), ": { ", quote_str("type"), ": ")
#
#   return(paste0(output,"}"))
# }

#' @export
#' @rdname tbl_lazy
simulate_vidar <- function() {
  structure(
    list(),
    class = c("ViDaRConnection", "DBIConnection")
  )
}
