# ========== ViDaR DBI Driver ========== #
# DBI Driver for ViDaR
setClass("ViDaRDriver", contains = "JDBCDriver")

# Instantiation of ViDaRDriver - using RJDBC
ViDaR <- function(driverClass="ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver", driverLocation="/home/sanca/sqlline/sqlline/bin/SQLPlanner-assembly-0.1.jar", quoteChar = "`") {
  tmp <- JDBC(driverClass, driverLocation, quoteChar)
  drv <- new ("ViDaRDriver")

  drv@jdrv <- tmp@jdrv
  drv@identifier.quote <- tmp@identifier.quote

  return(drv)
}


setMethod("dbGetInfo", "ViDaRDriver", def = function(dbObj, ...)
  list(name="ViDaRDriver", driver.version = utils::packageVersion("ViDaR"), DBI.version = utils::packageVersion("DBI"))
)

setMethod("dbIsValid", "ViDaRDriver", def = function(dbObj, ...) invisible(TRUE))

#
setMethod("dbConnect", "ViDaRDriver", def = function(drv, connectionString = "jdbc:pelago:model=/home/sanca/sqlline/sqlline/bin/schema.json", ...){

  jdbcconn <- RJDBC::dbConnect(as(drv,"JDBCDriver"), connectionString, ...)

  connenv <- new.env(parent = emptyenv())
  connenv$connectionString <- connectionString
  connenv$is_open <- TRUE

  conn <- new("ViDaRConnection", jc = jdbcconn@jc , identifier.quote = jdbcconn@identifier.quote, env=connenv)

  return(conn)
},
valueClass = "ViDaRConnection")



# ========== ViDaR DBI Connection ========== #

# DBI Connection for ViDaR - inherits JDBC connection
setClass("ViDaRConnection", contains = "JDBCConnection", slots = list(env="environment"))

#setMethod("dbDataType", signature(dbObj="ViDaRConnection", obj="ANY"), def = function(dbObj, obj, ...)
#  invisible(TRUE) # Data type conversion
#  )

setMethod("dbDisconnect", "ViDaRConnection", def = function(conn, ...){

  RJDBC::dbDisconnect(as(conn, "JDBCConnection"))
  conn@env$is_open <- FALSE
  })

setMethod("dbGetInfo", "ViDaRConnection", def = function(dbObj, ...){
  info = list(connectionString = dbObj@env$connectionString, identifierQuote = dbObj@identifier.quote)
  })

#setMethod("dbExistsTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
#    return(as.character(name) %in% dbListTables(conn))
#  })

#setMethod("dbGetException", "ViDaRConnection", def = function(conn, ...)
#  invisible(TRUE)
#  )

#setMethod("dbIsValid", "ViDaRConnection", def = function(dbObj, ...)
#  invisible(TRUE)
#  )

setMethod("dbListFields", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){

    if(dbExistsTable(conn,name)){

      metaData <- .jcall(conn@jc, "Ljava/sql/DatabaseMetaData;", "getMetaData", check=FALSE)
      resultSet <- .jcall(metaData, "Ljava/sql/ResultSet;", "getColumns",
                          .jnull("java/lang/String"), .jnull("java/lang/String"), name, "%", check=FALSE)

      fields <- c()

      while(.jcall(resultSet, "Z", "next")) {
        fields <- c(fields, .jcall(resultSet, "S", "getString", "COLUMN_NAME"))
      }

      return(fields)
    } else {
      stop(paste0("Table ", name, " does not exist."))
    }

  })

#setMethod("dbReadTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...){
#    if(!dbExistsTable(conn, name))
#      stop(paste("Table: ", name, " - does not exist"))

    ## this is the practical effect of invocation - just read the whole file
#    dbGetQuery(conn, paste0("SELECT * FROM ", name))

#  })

#setMethod("dbRemoveTable", signature(conn="ViDaRConnection", name="character"), def = function(conn, name, ...)
#  invisible(TRUE)
#  )

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
    env$table_name <- extractFrom(as.character(statement), conn@identifier.quote)

    return(new("ViDaRResult", env=env))
  }

  # send the query to ViDa for execution
  print('raw statement:')
  print(as.character(statement))
  print('sent statement:')
  print(textProcessQuery(as.character(statement), conn@identifier.quote))

  jdbcres <- RJDBC::dbSendQuery(as(conn, "JDBCConnection"), textProcessQuery(as.character(statement), conn@identifier.quote))

  env$conn <- conn
  env$query <- statement
  env$lazy <- FALSE

  invisible(new("ViDaRResult", jr=jdbcres@jr, md=jdbcres@md, stat=jdbcres@stat, pull=jdbcres@pull,  env=env))
  })

setMethod("dbSendStatement", signature(conn="ViDaRConnection", statement="character"), def = function(conn, statement, ...){

  RJDBC::dbSendUpdate(as(conn, "JDBCConnection"), textProcessQuery(statement))

})

setMethod("dbSendUpdate", signature(conn="ViDaRConnection", statement="character"), def = function(conn, statement, ...){

  RJDBC::dbSendUpdate(as(conn, "JDBCConnection"), textProcessQuery(statement))

})

#setMethod("dbWriteTable", signature(conn="ViDaRConnection", name="character", value="ANY"), def = function(conn, name, value, ...)
#  invisible(TRUE)
#  )

# JDBC consideres every query a transaction unless otherwise specified
setMethod("dbBegin", signature = (conn="ViDaRConnection"), def = function(conn, ...)
  invisible(TRUE)
  )

# ========== ViDaR DBI Result ========== #

# DBI Result for ViDaR
setClass("ViDaRResult", contains = "JDBCResult", slots = list(env="environment"))


# TODO: FIX THE ERROR WHEN res@jr is null
setMethod("dbClearResult", "ViDaRResult", def = function(res, ...) {

    if(!is.jnull(res@jr)) {
      .jcall(res@jr, "V", "close")
      .jcall(res@stat, "V", "close")
      invisible(TRUE)
    }

  })

#setMethod("dbColumnInfo", "ViDaRResult", def = function(res, ...)
#  invisible(TRUE)
#  )

setMethod("dbFetch", signature(res="ViDaRResult", n="numeric"), def = function(res, n=1, ...) {

    if(res@env$lazy){
      print("Lazy get table: ")
      print(res@env$table_name)

      return(schema2tbl(res@env$table_name, res@env$conn))
    } else {
      return(RJDBC::fetch(as(res, "JDBCResult"), n))
    }
  })
