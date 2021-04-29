# Contains utility functions intended for opening and creating a table from files (e.g. CSV, JSON)

# Utility function for getting the last n characters from a string
getLastChars <- function(string, n)
  return(substr(string, nchar(string)-n+1, nchar(string)))

# Simple wrapper for opening CSV files and retrieving a dataframe
# TODO: the difference is that we need to specify the fields and types, linehint...
# Fields - example: list(a="integer", b="varchar")
readcsv <- function(connection, fields = NULL, path, linehint = NULL, local = TRUE, name = NULL, remotePath = NULL,
                    sep = ',', header = TRUE, colClasses = NULL, colNames = NULL, lines = NULL, policy,
                    delimiter = NULL, brackets = TRUE) {
  return(readcsv2(file=path, nrows=nrows, fields, local, name, fromLocalFilePath = remotePath, sep, header, colClasses, colNames, policy, delimiter, brackets, conn=connection, policy=policy));
}

plugins.csv.create <- function(file, nrows, policy, sep, brackets, header){
  return(c(paste0("%plugin%:{",
   paste0("%lines%: ", toString(as.integer(nrows)), ","),
   paste0("%policy%: ", toString(as.integer(policy)), ","),
   if(!missing(sep)) paste0("%delimiter%: %", sep, "% ,"),
   if(!missing(brackets)) paste0("%brackets%: ", if(brackets) "true" else "false", ","),
   if(header) "%hasHeader%: true,",
   "%type%: %csv%",
   "}"),
    paste0("%file%: %", file, "%")))
}

vidar.new.table <- function(conn, name, fields, metadata){
  # We do not want to redefine(=create) tables
  knowntables <- conn@env$tables
  if (exists(name, envir = knowntables)) {
    d <- get(name, envir = knowntables)
    if (d[1] != metadata) {
      stop("Metadata mismatch with previous declaration")
    }
    return(d[2])
  } else {
    j <- paste0(" JPLUGIN `{", paste(metadata, collapse=", "), "}`")
    dbCreateTable(conn = conn, name = name, fields = fields, plugin.metadata = j)
    t <- tbl(conn, name)
    assign(name, c(metadata, t), envir = knowntables)
    return(t)
  }
}

readcsv2 <- function(file, header = TRUE, sep = ',', quote = "\"", 
                    col.names, nrows, fields = NULL, colClasses = NA, policy, fromLocalFilePath, name,
                    conn) {
  brackets <- quote != ""
  if(is.null(nrows) || nrows < 0){
    stop("(Exact) number of lines in file required");
  }

  # if name is not specified, extract it from path
  if(missing(name)) name = file

  # # if fields are specified
  # if(!is.null(fields)){
  #   # if fields are specified as string, consider it is as col_name:col_type list
  #   if(is.character(fields))
  #     fields = string2list(fields)
  # } else {

  # read fields as dataframe and use that information, if column classes are not defined
  if(missing(colClasses)) {
    if(missing(col.names))
      fields <- utils::read.csv(file, sep = sep, nrows = 10, header = header)
    else
      fields <- utils::read.csv(file, sep = sep, nrows = 10, header = header, col.names = col.names)
  } else {
    # if column names are undefined, then generate them, else use the ones in col.names list
    if(missing(col.names) || is.null(col.names)){
      col.names <- paste0("V", as.character(c(1:length(colClasses))))
    }
    if(length(col.names)!=length(colClasses))
      stop("col.names and colClasses must be of same length")

    # create list of fields from col.names and colClasses
    fields <- string2list(paste0(col.names, ":", colClasses, collapse = ","), delimiter = ":")
  }

  # }

  if (missing(policy)) {
    policy <- length(fields) %/% 2
  }

  if (policy < 0) stop("Missing or invalid policy");

  if(!missing(fromLocalFilePath)){
    stop("TODO: Add here logic to transfer the file to the remote engine")
    file <- fromLocalFilePath
  }

  metadata <- plugins.csv.create(file = file, nrows = nrows, policy = policy, sep = sep, brackets = brackets, header = header)

  return(vidar.new.table(conn, name=name, fields=fields, metadata=metadata))
}

vidar.default.connection <- function(conn){
  name <- ".vidar.default.connection"
  env <- environment(vidar.default.connection)
  if (!missing(conn)){
    # assign(name, conn, envir = env)
  } else {
    if (!exists(name, envir = env)){
      return(NULL)
    }
    return(get(name, envir = env)) 
  }
}

read.csv <- function(file, ..., policy = NULL, conn) {
  if (missing(conn)) conn <- vidar.default.connection()
  if (missing(conn) || is.null(conn)){
    utils:::read.csv(file, ...)
  } else {
    print("Using ViDaR function over R's default")
    readcsv2(file, ..., policy = policy, conn=conn)
  }
}

plugins.json.create <- function(file, nrows, policy){
  return(c(paste0(
   "%plugin%:{",
   paste0("%lines%: ", toString(as.integer(nrows)), ","),
   paste0("%policy%: ", toString(as.integer(policy)), ","),
   "%type%: %json%",
   "}"),
   paste0("%file%: %", file, "%")))
}

readjson2 <- function(connection, name, json, json_quote="\"") {
  json_sub <- gsub(json_quote, "%", json)
  query <- SQL(paste0("CREATE TABLE ", name, " FROM_JSON `", json_sub, "`"))
  dbSendUpdate(con, query)
  return(tbl(connection, name))
}


read.json.reconstruct <- function(file, name, fields, metadata){
  return(paste0('{"', name, '": ',
    '{"path": "inputs/json/employees-flat.json", ',
    '"type": ', fields, ', ',
    paste(metadata, collapse=", "), '} }'))
}

readjson <- function(file, ..., fields, nrows = -1, policy, name, conn) {
  # if name is not specified, extract it from path
  if(missing(name)) name = file

  if (missing(policy)) {
    print(fields)
    policy <- stringi::stri_count(fields, fixed="attrName") %/% 2
    print(policy)
  }

  if (policy < 0) stop("Missing or invalid policy");

  metadata <- plugins.json.create(file = file, nrows = nrows, policy = policy)
  # return(vidar.new.table(conn, name=name, fields=ffffffff, metadata=metadata))

  # We do not want to redefine(=create) tables
  knowntables <- conn@env$tables
  if (exists(name, envir = knowntables)) {
    d <- get(name, envir = knowntables)
    if (d[1] != metadata) {
      stop("Metadata mismatch with previous declaration")
    }
    return(d[2])
  } else {
    json_sub <- read.json.reconstruct(file=file, name=name, fields=fields, metadata=metadata)
    query <- SQL(paste0("CREATE TABLE ", dbQuoteIdentifier(conn, name), " FROM_JSON `", json_sub, "`"))
    dbSendUpdate(conn, query)
    t <- tbl(conn, name)
    assign(name, c(metadata, t), envir = knowntables)
    return(t)
  }
}

# ndjson == \n delimited JSON
fromJSON <- function(file, ..., nrows = -1, fields = NULL, ndjson=FALSE, policy, conn) {
  if (missing(conn)) conn <- vidar.default.connection()
  if (missing(conn) || is.null(conn)){
    if (ndjson) {
      jsonlite::stream_in(file(file))
    } else {
      jsonlite::fromJSON(file, ...)
    }
  } else {
    print("Using ViDaR function over R's default")
    readjson(file, ..., fields=fields, nrows = nrows, policy = policy, conn=conn)
  }
}

read.json <- fromJSON

# in format name1:type1, name2:type2...
string2list <- function(str, delimiter = ":") {

  # quote the strings - a:b -> 'a':'b'
  repl <- str
  repl <- gsub(paste0(delimiter, "([a-zA-z._0-9]+)"),"='\\1'", repl)
  repl <- gsub(delimiter, "=", repl)

  return(lazyeval::lazy_eval(paste0("list(",repl,")")))
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
