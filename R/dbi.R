# ========== ViDaR DBI Driver ========== #
# --- don't forget to load library(DBI) ---
# DBI Driver for ViDaR
setClass("ViDaRDriver", contains = "DBIDriver")

# Instantiation of ViDaRDriver
ViDaR <- function() new ("ViDaRDriver")

# Overloading dbGetInfo
setMethod("dbGetInfo", "ViDaRDriver", def = function(dbObj, ...)

  list(name="ViDaRDriver", driver.version = utils::packageVersion("ViDaR"), DBI.version = utils::packageVersion("DBI"))

  )

# Overloading dbIsValid
setMethod("dbIsValid", "ViDaRDriver", def = function(dbObj, ...) invisible(TRUE))

# Overloading dbUnloadDriver
setMethod("dbUnloadDriver", "ViDaRDriver", def = function(drv, ...) invisible(TRUE))


# Overloading dbConnect
setMethod("dbConnect", "ViDaRDriver", def = function(drv, dbhost="localhost", dbport=50001, ...){

  connenv <- new.env(parent = emptyenv())
  connenv$host <- dbhost
  connenv$port <- dbport

  # establish socket connection and check connectivity - blocking set to true, to wait for execution
  # in DBI it is specified that it has to be sequential execution!
  tryCatch(
   {
      sockcon <- socketConnection(host=dbhost, port=dbport, blocking = TRUE, timeout=1234)
      response <- readLines(sockcon, 1)

      print("response:")
      print(response)
      connenv$conn <- sockcon
      connenv$is_open <- TRUE
    },
   error=function(cond){
      message("Cannot establish connection for given parameters.")
      #message(cond)
      connenv$is_open <- FALSE
    }
  )

  conn <- new("ViDaRConnection", env=connenv)
  return(conn)
},
valueClass = "ViDaRConnection")



# ========== ViDaR DBI Connection ========== #

# DBI Connection for ViDaR
setClass("ViDaRConnection", contains = "DBIConnection", slots = list(env="environment"))

#setMethod("dbDataType", signature(dbObj="ViDaRConnection", obj="ANY"), def = function(dbObj, obj, ...)
#  invisible(TRUE) # Data type conversion
#  )

setMethod("dbDisconnect", "ViDaRConnection", def = function(conn, ...){

  # send termination command for now
  writeLines("exit", conn@env$conn)

  # close socket connection and set flag
  close(conn@env$conn)
  conn@env$is_open <- FALSE
  conn@env$host <- NULL
  conn@env$port <- NULL

  invisible(TRUE)
  })

setMethod("dbGetInfo", "ViDaRConnection", def = function(dbObj, ...){
  #info = list()
  #info$host <- dbObj@env$host
  #info$port <- dbObj@env$port
  invisible(TRUE)
  #return(info)
  })

setMethod("dbExistsTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    return(as.character(name) %in% dbListTables(conn))
  })

setMethod("dbGetException", "ViDaRConnection", def = function(conn, ...)
  invisible(TRUE)
  )

setMethod("dbIsValid", "ViDaRConnection", def = function(dbObj, ...)
  invisible(TRUE)
  )

setMethod("dbListFields", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    if(dbExistsTable(conn,name)){
      if(conn@env$is_open){
        writeLines(paste0("list fields ", name), conn@env$conn)
        return(jsonlite::fromJSON(readLines(conn@env$conn,1)))
      }
    }
  })

setMethod("dbListTables", "ViDaRConnection", def = function(conn, ...) {
    if(conn@env$is_open){
      writeLines("list tables", conn@env$conn)
      return(jsonlite::fromJSON(readLines(conn@env$conn,1)))
    }
  })

setMethod("dbReadTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    if(!dbExistsTable(conn, name))
      stop(paste("Table: ", name, " - does not exist"))

    ## this is the practical effect of invocation - just read the whole file
    dbGetQuery(conn, paste0("SELECT * FROM ", name))

  })

setMethod("dbRemoveTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...)
  invisible(TRUE)
  )

processQuery <- function(query) {
  ret_query <- gsub("<SQL>", "", query)
  ret_query <- gsub("\"","", ret_query)
  ret_query <- gsub("LIMIT 10", "", ret_query)
  ret_query <- gsub("\n"," ", ret_query)
  ret_query <- gsub("\\(\\)","\\(*\\)", ret_query)

  return(ret_query)
}

extractFrom <- function(query) {
  from <- strsplit(processQuery(query), "FROM ")[[1]][2]
  from <- strsplit(from, " ")[[1]][1]
  return(from)
}

setMethod("dbSendQuery", signature(conn="ViDaRConnection", statement="character"), def = function(conn, statement, ...){
  # environment for ViDaRResult to return
  env <- new.env(parent = emptyenv())

  # case for loading 0 rows
  if(grepl("(0 = 1)",as.character(statement))){
    print('raw statement:')
    print(as.character(statement))

    env$conn <- conn
    env$query <- statement
    env$lazy <- TRUE
    env$table_name <- extractFrom(as.character(statement))

    return(new("ViDaRResult", env=env))
  }

  # send the query to ViDa for execution
  if(conn@env$is_open){

    print('raw statement:')
    print(as.character(statement))

    writeLines(processQuery(as.character(statement)), conn@env$conn)

    print('sent statement:')
    print(processQuery(as.character(statement)))

    response <- readLines(conn@env$conn, 1)
    env$response = response

    if(startsWith(response, "result in file")){
      env$success = TRUE
      # for now parse the message 'result in file PATH' and save it in the environment
      env$path = strsplit(response, "result in file ")[[1]][2]
    } else {
      env$success = FALSE
    }

  } else {
    env$success = FALSE
  }

  env$conn <- conn
  env$query <- statement
  env$lazy <- FALSE

  invisible(new("ViDaRResult", env=env))
  })

setMethod("dbWriteTable", signature(conn="ViDaRConnection", name="character", value="ANY"), def = function(conn, name, value, ...)
  invisible(TRUE)
  )

setMethod("dbBegin", signature = (conn="ViDaRConnection"), def = function(conn, ...)
  invisible(TRUE)
  )

#setMethod("dbBegin", signature = (conn="ViDaRConnection"), def = function(conn, ...)
#  invisible(TRUE)
#)

# ========== ViDaR DBI Result ========== #

# DBI Result for ViDaR
setClass("ViDaRResult", contains = "DBIResult", slots = list(env="environment"))

setMethod("dbClearResult", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbColumnInfo", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

# table name - find path
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

  # TEST PURPOSES
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

setMethod("dbFetch", signature(res="ViDaRResult", n="numeric"), def = function(res, n=1, ...) {

  if(res@env$lazy){
    print("Lazy get table: ")
    print(res@env$table_name)
    return(schema2tbl(res@env$table_name))
  } else {

    fetch_query <- paste0("fetch ",res@env$path, " ", as.character(n))
    writeLines(fetch_query, res@env$conn@env$conn)
    json_res <- jsonlite::fromJSON(readLines(res@env$conn@env$conn,1))

    tryCatch({
      #return(as.tbl(jsonlite::fromJSON(res@env$path)))
      return(as.tbl(json_res))
    },
    error=function(cond){
      # TODO: better handling of empty result
      #return(as.tbl(as.data.frame(jsonlite::fromJSON(res@env$path))))
      return(as.tbl(as.data.frame(json_res)))
    })

  }

  #result_file <- file(res@env$resp_path)
  #print("Result:")
  #print(readLines(result_file))

  #schema <- jsonlite::read_json(

  # THIS HAS TO BE MORE ELEGANT AND FASTER (MAYBE RCPP)
  #tryCatch(
  #  {
      #return(as.tbl(jsonlite::fromJSON(res@env$resp_path)))
  #  },
  #  error=function(cond){
  #   return(as.tbl(jsonlite::fromJSON(res@env$path)))
  #  }
  #)

  ### NEW - make data frame out of CSV schema
  # s <- read.csv("customer.csv")
  #
  # build_cmd <- "list("
  # type_map <- list(int="integer(0)", string="character(0)")
  #
  # for(col in colnames(s)){
  #
  #   name <- strsplit(col,"\\.")[[1]][1]
  #   type <- strsplit(col,"\\.")[[1]][2]
  #
  #   build_cmd <- paste0(build_cmd, name, "=", type_map[type], ",")
  # }
  #
  # build_cmd<-substr(build_cmd, 1, nchar(build_cmd)-1)
  # build_cmd<-paste0(build_cmd,")")
  #
  # df <- data.frame(lazyeval::lazy_eval(build_cmd))

  })

setMethod("dbGetRowCount", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbGetRowsAffected", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbGetStatement", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbHasCompleted", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbIsValid", "ViDaRResult", def = function(dbObj, ...)
  invisible(TRUE)
  )
