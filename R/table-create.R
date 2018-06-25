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
setMethod("sqlCreateTable", signature("ViDaRConnection"),
          function(con, table, fields,  temporary = FALSE, path = NULL, type = NULL, linehint = NULL, ...) {

            table <- dbQuoteIdentifier(con, table)

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
                "  ", paste(fields, collapse = ",\n  "), "\n)\n"#, "PATH: ", path, ",\n", "METADATA: ", adapter, "\n"
              ))
            }

            metadata <- ""
            if(!is.null(path) || !is.null(type) || !is.null(linehint)){
              metadata <- paste0(metadata, " JPLUGIN `{'plugin':{'type': '", if(!is.null(type)) type else "block", "',",
                                 "'linehint': ", if(!is.null(linehint)) toString(as.integer(linehint)) else "20000", "}, ",
                                 "'file': '", if(!is.null(type)) type else "default_path", "'}`")
            }

            query <- SQL(paste0(query, metadata))

            return(query)
          }
)

setMethod("dbCreateTable", signature("ViDaRConnection"),
          def = function(conn, name, fields, ..., path = NULL, type = NULL, linehint = NULL, row.names = NULL, temporary = FALSE) {

            query <- sqlCreateTable(
              con = conn,
              table = name,
              fields = fields,
              temporary = temporary,
              path = path,
              type = type,
              linehint = linehint,
              ...
            )

            dbSendUpdate(conn, query)
            invisible(TRUE)

          })

