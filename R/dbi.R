# ========== ViDaR DBI Driver ========== #
# --- don't forget to load library(DBI) ---
# DBI Driver for ViDaR
setClass("ViDaRDriver", contains = "DBIDriver")

# Instantiation of ViDaRDriver
ViDaR <- function() new ("ViDaRDriver")


# Overloading dbIsValid
setMethod("dbIsValid", "ViDaRDriver", def = function(dbObj, ...) invisible(TRUE))

# Overloading dbUnloadDriver
setMethod("dbUnloadDriver", "ViDaRDriver", def = function(drv, ...) invisible(TRUE))

# Overloading dbGetInfo
setMethod("dbGetInfo", "ViDaRDriver", def = function(dbObj, ...)

  list(name="ViDaRDriver", driver.version = utils::packageVersion("ViDaR"), DBI.version = utils::packageVersion("DBI"))

  )

# Overloading dbConnect
setMethod("dbConnect", "ViDaRDriver", def = function(drv, vida_sh_path="~/Desktop/vida.sh", schema_path, schema_name, tmp_folder, query_out, ...){

  connenv <- new.env(parent = emptyenv())
  connenv$sh_path <- vida_sh_path
  connenv$schema_name <- schema_name
  connenv$schema_path <- schema_path
  connenv$tmp_folder <- tmp_folder
  connenv$query_output <- query_out
  connenv$is_open <- TRUE

  # create a simple query to generate the schema
  system2(command = "sh", args = paste(vida_sh_path, " ", tmp_folder, "tmp.sql", " ", schema_name, sep=""), stdout = FALSE, stderr = FALSE)

  conn <- new("ViDaRConnection", env=connenv)
  return(conn)
},
valueClass = "ViDaRConnection")




# ========== ViDaR DBI Connection ========== #

# DBI Connection for ViDaR
setClass("ViDaRConnection", contains = "DBIConnection", slots = list(env="environment"))

setMethod("dbDataType", signature(dbObj="ViDaRConnection", obj="ANY"), def = function(dbObj, obj, ...)
  invisible(TRUE) # Data type conversion
  )

setMethod("dbDisconnect", "ViDaRConnection", def = function(conn, ...)
  invisible(TRUE)
  )

setMethod("dbGetInfo", "ViDaRConnection", def = function(dbObj, ...)
  invisible(TRUE) # Connection info
  )

setMethod("dbExistsTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    return(as.character(name) %in% dbListTables(conn))
  }
  )

setMethod("dbGetException", "ViDaRConnection", def = function(conn, ...)
  invisible(TRUE)
  )

setMethod("dbIsValid", "ViDaRConnection", def = function(dbObj, ...)
  invisible(TRUE)
  )

setMethod("dbListFields", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    schema <- jsonlite::read_json(paste(conn@env$schema_path, conn@env$schema_name, sep = ""))

    fields <- schema[[name]]$type$inner$attributes

    list_fields <- list()
    for(field in fields){
      list_fields <- c(list_fields, field$attrName)
    }

    return(as.character(list_fields))
  }
  )

setMethod("dbListTables", "ViDaRConnection", def = function(conn, ...) {
    # reads the JSON from schema
    schema <- jsonlite::read_json(paste(conn@env$schema_path, conn@env$schema_name, sep = ""))
    return(names(schema))
  })

setMethod("dbReadTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
    if(!dbExistsTable(conn, name))
      stop(paste("Table: ", name, " - does not exist"))

    # select * from workaround - concat the list of fields with ','
    fields<-paste(dbListFields(con, name),collapse=',')
    dbGetQuery(conn, paste0("SELECT ",fields," FROM ", name, ";"))
  }
  )

setMethod("dbRemoveTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...)
  invisible(TRUE)
  )

# process the query to suit ViDa
processQuery <- function(query, con, ...) {

  ret <- tolower(query)

  ret <- gsub('<sql> ','',ret)
  ret <- gsub('from','\nfrom',ret)
  ret <- gsub('where','\nwhere', ret)
  ret <- gsub(';','\n;', ret)

  #split <- gsub('"','',regmatches(ret, gregexpr("(?<=\")(.*?)(?=\")", ret, perl = TRUE))) # split to extract the table name from first quotes

  ret <- gsub('\\*',paste(dbListFields(con, 'part'),collapse=','), ret)

  ret <- gsub("\"","", ret)
  ret <- gsub("where [(]0 = 1[)]","",ret)

  # UNQOUTE!
  ret <- gsub("limit 10","",ret)

  if(!grepl(';',ret))
    ret <- paste0(ret,'\n;')

  ret <- gsub('\n\n','\n', ret)
  #ret <- gsub('\n;',';', ret)

  # REMOVE PARENTHESES
  ret <- gsub("\\(", "", ret)
  ret <- gsub("\\)", "", ret)

  return(ret)
}

setMethod("dbSendQuery", signature(conn="ViDaRConnection", statement="character"), def = function(conn, statement, ...){
  env <- new.env(parent = emptyenv())

  print("Query:")
  print(statement)

  statement<-processQuery(statement, conn)

  print("Processed query:")
  print(statement)

  # create a file as vida query input
  query_file <- file(paste0(conn@env$tmp_folder,'tmp.sql'))
  writeLines(statement, query_file)
  close(query_file)

  # send query to vida - old vesion, which printed the whole output to a file
  # out <- system2(command = "sh", args = paste(conn@env$sh_path, " ", conn@env$tmp_folder, "tmp.sql", " ", conn@env$schema_name, sep=""), stdout = TRUE, stderr = FALSE)
  # CURRENT VERSION - WRITES TO A SPECIFIC FILE
  system2(command = "sh", args = paste(conn@env$sh_path, " ", conn@env$tmp_folder, "tmp.sql", " ", conn@env$schema_name, sep=""), stdout = FALSE, stderr = FALSE)
  # write output to file
  #out_file <- file(paste0(conn@env$tmp_folder,'tmp.txt'))
  #writeLines(out, out_file)
  #close(out_file)

  env$success = TRUE
  env$conn <- conn
  env$query <- statement
  env$resp_path <- conn@env$query_output

  invisible(new("ViDaRResult", env=env))
  })

setMethod("dbWriteTable", signature(conn="ViDaRConnection", name="character", value="ANY"), def = function(conn, name, value, ...)
  invisible(TRUE)
  )




# ========== ViDaR DBI Result ========== #

# DBI Result for ViDaR
setClass("ViDaRResult", contains = "DBIResult", slots = list(env="environment"))

setMethod("dbClearResult", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbColumnInfo", "ViDaRResult", def = function(res, ...)
  invisible(TRUE)
  )

setMethod("dbFetch", signature(res="ViDaRResult", n="numeric"), def = function(res, n, ...) {

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
     return(as.data.frame(jsonlite::fromJSON(res@env$resp_path)))
  #  }
  #)
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
