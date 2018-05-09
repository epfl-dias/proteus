setMethod("dbCreateTable", signature("ViDaRConnection"),
          def = function(conn, name, fields, ..., temporary = FALSE, path) {

            query <- sqlViDaCreateTable(
              con = conn,
              table = name,
              fields = fields,
              temporary = temporary,
              path = path,
              ...
            )

            dbExecute(conn, query)
            invisible(TRUE)

          })

setMethod("sqlViDaCreateTable", signature("ViDaRConnection"),
          function(con, table, fields,  temporary = FALSE, path, ...) {

            table <- dbQuoteIdentifier(con, table)

            # Cover the JSON case and add the information about path and/or other options.
            # In general JSON is a nested list - so recursive calls will be necessary while building
            # the SQL string. On each level a keyword will be added indicating that we are entering
            # a nested level

            if (is.data.frame(fields)) {
              fields <- sqlRownamesToColumn(fields, NULL)
              fields <- vapply(fields, function(x) DBI::dbDataType(con, x), character(1))
            }

            field_names <- dbQuoteIdentifier(con, names(fields))
            field_types <- unname(fields)
            fields <- paste0(field_names, " ", field_types)

            SQL(paste0(
              "CREATE ", if (temporary) "TEMPORARY ", "TABLE ", table, " (\n",
              "  ", paste(fields, collapse = ",\n  "), "\n)\n"
            ))
          }
)
