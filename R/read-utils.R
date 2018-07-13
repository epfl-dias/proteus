# Contains utility functions intended for opening and creating a table from files (e.g. CSV, JSON)

# Utility function for getting the last n characters from a string
getLastChars <- function(string, n)
  return(substr(string, nchar(string)-n+1, nchar(string)))

# Simple wrapper for opening CSV files and retrieving a dataframe
# TODO: the difference is that we need to specify the fields and types, linehint...
# Fields - example: list(a="integer", b="varchar")
readcsv <- function(connection, fields = NULL, path, linehint = NULL, local = TRUE, name = NULL, remotePath = NULL,
                    sep = ',', header = TRUE, colClasses = NULL, colNames = NULL, lines = NULL, policy = NULL,
                    delimiter = NULL, brackets = NULL) {
  if(is.null(path))
    stop("Path cannot be undefined")

  if(is.null(linehint) && is.null(lines))
    stop("Either linehint or lines has to be defined")

  # if name is not specified, extract it from path
  if(is.null(name))
    name = getLastChars(path,10)

  # if fields are specified
  if(!is.null(fields)){
    # if fields are specified as string, consider it is as col_name:col_type list
    if(is.character(fields))
      fields = string2list(fields)
  } else {
    # read fields as dataframe and use that information, if column classes are not defined
    if(is.null(colClasses)) {
      if(is.null(colNames))
        fields <-read.csv(path, sep = sep, nrows = 10, header = header)
      else
        fields <-read.csv(path, sep = sep, nrows = 10, header = header, col.names = colNames)
    }
    else {
      # if column names are undefined, then generate them, else use the ones in colNames list
      if(is.null(colNames)){
        colNames <- paste0("V", as.character(c(1:length(colClasses))))
        fields <- string2list(paste0(colNames, ":", colClasses, collapse = ","), delimiter = ":")
      } else {
        if(length(colNames)!=length(colClasses))
          stop("colNames and colClasses must be of same length")

        # create list of fields from colNames and colClasses
        fields <- string2list(paste0(colNames, ":", colClasses, collapse = ","), delimiter = ":")
      }
    }

  }

  if(!local){
    if(is.null(remotePath))
      stop("remote path has to be defined")
    else{
      print("Some logic to transfer the file to a remote")

      dbCreateTable(conn = con, name = name, fields = fields, path = remotePath, linehint = linehint, lines = lines, type = "csv",
                    policy = policy, delimiter = delimiter, brackets = brackets)
    }
  } else {
    dbCreateTable(conn = con, name = name, fields = fields, path = path, linehint = linehint, lines = lines, type = "csv",
                  policy = policy, delimiter = delimiter, brackets = brackets)
  }

  return(tbl(connection, name))
}

# in format name1:type1, name2:type2...
string2list <- function(str, delimiter = ":") {

  # quote the strings - a:b -> 'a':'b'
  repl <- gsub("([a-zA-z._0-9]+)","'\\1'", str)
  repl <- gsub(delimiter, "=", repl)

  return(lazyeval::lazy_eval(paste0("list(",repl,")")))
}

readjson2 <- function(connection, name, json, json_quote="\"") {
  json_sub <- gsub(json_quote, "%", json)

  query <- SQL(paste0("CREATE TABLE ", name, " FROM_JSON `", json_sub, "`"))
  dbSendUpdate(con, query)
  return(tbl(connection, name))
}

# TODO - reading JSON from fields, which is of type list (nesting is achieved with lists of lists)
# readjson <- function(connection, fields, path, linehint, local = TRUE, name = NULL, remotePath = NULL) {
#
#   if(is.null(path))
#     stop("Path cannot be undefined")
#
#   if(is.null(linehint))
#     stop("Linehing cannot be undefined")
#
#   if(is.null(name))
#     name = getLastChars(path,10)
#
#   # if fields are specified
#   if(!is.null(fields)){
#     # if fields are specified as string, consider it is as col_name:col_type list
#     if(is.character(fields))
#       fields = string2list(fields)
#   } else {
#     stop("Fields cannot be undefined")
#   }
#
#   # TODO - generating JSON from the string
#   #json <- ...
#
#   if(!local){
#     if(is.null(remotePath))
#       stop("remote path has to be defined")
#     else{
#       print("Some logic to transfer the file to a remote")
#
#       json_sub <- gsub(json_quote, "%", json)
#
#       query <- SQL(paste0("CREATE TABLE ", name, " FROM_JSON `", json_sub, "`"))
#       dbSendUpdate(con, query)
#     }
#   } else {
#
#     json_sub <- gsub(json_quote, "%", json)
#
#     query <- SQL(paste0("CREATE TABLE ", name, " FROM_JSON `", json_sub, "`"))
#     dbSendUpdate(con, query)
#   }
#
#
#   return(tbl(connection, name))
# }
