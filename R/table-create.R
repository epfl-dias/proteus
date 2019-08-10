lazy_eval_type <- function(con, type) {
  DBI::dbDataType(con, lazyeval::lazy_eval(type))
}

create_table_level <- function(list, parent_name){

  query <- paste0(sqlRownamesToColumn(parent_name, NULL), " RECORD TYPE \n (")

  for(el in names(list)){

    if(typeof(list[[el]])=="list"){
      query <- paste0(query, create_table_level(list[[el]], el), ",\n")
    } else {
      query <- paste0(query, " ", sqlRownamesToColumn(el, NULL),
                      "  ", lazy_eval_type(con, type_map[[list[[el]]]]), ",\n")
    }

  }

  # remove last comma
  query <- gsub(",\n$", "\n", query)

  query <- paste0(query, ")\n")

  return(query)
}

# jplugin `{\'plugin\':{ \'type\':\'block\', \'linehint\':200000 }, \'file\':\'/inputs/csv.csv\'}`
setMethod("dbQuoteIdentifier", signature("ViDaRConnection", "character"),
  function(conn, x){
    return(ifelse(grepl("^([A-Za-z][A-Za-z0-9_]*)?$", x), x, paste0("`", x, "`")))
  }
)

# jplugin `{\'plugin\':{ \'type\':\'block\', \'linehint\':200000 }, \'file\':\'/inputs/csv.csv\'}`
setMethod("sqlCreateTable", signature("ViDaRConnection"),
          function(con, table, fields,  temporary = FALSE, path = NULL, type = NULL, linehint = NULL,
                   lines = NULL, policy= NULL, delimiter = NULL, brackets = NULL, hasHeader = FALSE,
                   ...) {

            table <- dbQuoteIdentifier(con, table)
            print(table)

            # TODO: Cover the JSON case and add the information about path and/or other options.
            # In general JSON is a nested list - so recursive calls will be necessary while building
            # the SQL string. On each level a keyword will be added indicating that we are entering
            # a nested level
            if(is.list(fields) & !is.data.frame(fields)){

              query <- paste0("CREATE ", if(temporary) "TEMPORARY ", "TABLE ", table, " (\n ")

              for(el in names(fields)){
                if(typeof(fields[[el]])=="list"){
                  query <- paste0(query, " ", create_table_level(fields[[el]], el), ",\n")
                } else {
                  query <- paste0(query, " ", sqlRownamesToColumn(el, NULL),
                                  "  ", lazy_eval_type(con, type_map[[fields[[el]]]]), ",\n")
                }

              }

              # remove last comma
              query <- gsub(",\n$", "\n", query)
              query <- SQL(paste0(query, "\n)\n"))

            } else {

              if (is.data.frame(fields)) {
                fields <- sqlRownamesToColumn(fields, NULL)
                fields <- vapply(fields, function(x) DBI::dbDataType(con, x), character(1))
              }

              field_names <- dbQuoteIdentifier(con, names(fields))
              field_types <- unname(fields)
              fields <- paste0(field_names, " ", field_types)

              query <- SQL(paste0(
                "CREATE ", if (temporary) "TEMPORARY ", "TABLE ", table, " (\n",
                "  ", paste(fields, collapse = ",\n  "), "\n)\n"
              ))
            }

            # appending plugin info - % will be replaced by \\\" in query text processing
            metadata <- ""
            if(!is.null(path) || !is.null(type) || !is.null(linehint) || !is.null(lines) || !is.null(policy) || !is.null(delimiter)
               || !is.null(brackets) || hasHeader){
              metadata <- paste0(metadata, " JPLUGIN `{%plugin%:{",
                                 if(!is.null(lines)) paste0("%lines%: ", toString(as.integer(lines)), ","),
                                 if(!is.null(linehint)) paste0("%linehint%: ", toString(as.integer(linehint)), ","),
                                 if(!is.null(policy)) paste0("%policy%: ", toString(as.integer(policy)), ","),
                                 if(!is.null(delimiter)) paste0("%delimiter%: %", delimiter, "% ,"),
                                 if(!is.null(brackets)) paste0("%brackets%: ", if(brackets) "true" else "false", ","),
                                 if(hasHeader) "%hasHeader%: true,",

                                 "%type%: %", if(!is.null(type)) type else "block", "%",
                                 "}, ",
                                 "%file%: %", if(!is.null(path)) path else "default_path", "%}`")
            }

            query <- SQL(paste0(query, metadata))

            return(query)
          }
)

setMethod("dbCreateTable", signature("ViDaRConnection"),
          def = function(conn, name, fields, ..., path = NULL, type = NULL, linehint = NULL, lines = NULL, policy = NULL, delimiter = NULL, brackets = NULL, hasHeader = FALSE,
                         row.names = NULL, temporary = FALSE # rownames and temporary always in the end per contract
                         ) {

            query <- sqlCreateTable(
              con = conn,
              table = name,
              fields = fields,
              temporary = temporary,
              path = path,
              type = type,
              linehint = linehint,
              lines = lines,
              policy = policy,
              delimiter = delimiter,
              brackets = brackets,
              hasHeader = hasHeader,
              ...
            )

            dbSendUpdate(conn, query)
            invisible(TRUE)

          })

