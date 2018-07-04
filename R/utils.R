# DBI Utils

# Utility function for preprocessing the text of the query -
# escaped quotes are deletd from the query
# WARNING: MAKE SURE THE QUERY IS PREPROCESSED ONLY ONCE!
textProcessQuery <- function(query, quoteChar = "`") {
  ret_query <- gsub("\"", quoteChar, query)
  ret_query <- gsub("%", "\\\"", ret_query)
  ret_query <- gsub("\\\n", "", ret_query)
  ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)
  #ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)
  ret_query <- gsub("LIMIT 0", "", ret_query)

  #TODO: temporarly disable
  return(ret_query)
  #return(query)
}

# Util function for extracting the table name in FROM clause
extractFrom <- function(query, quoteChar = "`") {
  from <- strsplit(textProcessQuery(query, quoteChar), "FROM ")[[1]][2]
  from <- strsplit(from, " ")[[1]][1]
  from <- gsub("`", "", from)
  return(from)
}

# for case of creating a tbl (return 0 rows), R magic with lazy evaluation
# WILL NEED TO COVER THE CASE FOR NON-STANDARD TYPES (e.g. generate dataFrame first, then flatten it if nested types are present)
schema2tbl <- function(table, con){

  # TEST PURPOSES FOR NESTED SCHEMAS
  if(table=="emp") {
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
      build_cmd <- c(build_cmd, parseRecordType(ct, cn))

    colName <- c(colName, cn)
    colType <- c(colType, ct)
  }

  cmd <- paste0("data.frame(", paste(build_cmd, collapse = ","), ")")

  on.exit(.jcall(resultSet, "V", "close"))

  return(as.tbl(lazyeval::lazy_eval(cmd)))
}

#' @export
#' @rdname tbl_lazy
simulate_vidar <- function() {
  structure(
    list(),
    class = c("ViDaRConnection", "DBIConnection")
  )
}
