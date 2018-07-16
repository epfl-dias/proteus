for_each <- function(.data, ...) {
  dots <- quos(...)
  add_op_single("for_each", .data, dots = ...)
}


sql_clause_unnest <- function(from, unnest, con){

  unnest <- gsub("~", "", as.character(unnest))

  if (length(from) > 0L && length(unnest) > 0L) {
    assertthat::assert_that(is.character(from))
    assertthat::assert_that(is.character(unnest))

    from_name <- random_table_name()

    build_sql(
      sql("FROM"), " ",
      sql(paste(from, from_name)),
      sql(", UNNEST ("),
      sql(paste0(from_name, ".", ident(unnest), collapse = ", ")),
      ") "
    )
  }
}


sql_unnest <- function(con, select, from, unnest, ...) {

  out <- vector("list", 3)
  names(out) <- c("select", "from", "unnest")

  out$select <- dbplyr:::sql_clause_select(select, con, distinct = FALSE)

  out$from <- sql_clause_unnest(from, unnest, con)

  dbplyr::escape(unname(dbplyr:::compact(out)), collapse = " ", parens = FALSE, con = con)
}


sql_build.op_for_each <- function(op, con, ...) {
  # modify select query to accept unnest parameter
  # create a new function, similar to select query
  # or extend select query with new parameter
  # references:
  # tbl-lazy, sql-render, sql-query, sql-generic, sql-build

  unnest_query(sql_build(op$x, con), unnest = op$dots)
}


sql_render.unnest_query <- function(query, con = NULL, ..., root = FALSE) {
  from <- sql_subquery(con, sql_render(query$from, con, ..., root = root), name = NULL)

  sql_unnest(
    con, query$select, from, query$unnest, where = query$where, group_by = query$group_by,
    having = query$having, order_by = query$order_by, limit = query$limit,
    distinct = query$distinct,
    ...
  )
}

unnest_query <- function(from,
                         select = sql("*"),
                         unnest = character()) {
  structure(
    list(
      from = from,
      select = select,
      unnest = unnest
    ),
    class = c("unnest_query", "query")
  )
}

print.unnest_query <- function(x, ...) {
  cat(
    "<SQL SELECT",
    sep = ""
  )
  cat("From:     ", gsub("\n", " ", sql_render(x$from, root = FALSE)), "\n", sep = "")

  if (length(x$unnest))   cat("Unnest:   ", named_commas(x$unnest), "\n", sep = "")
  if (length(x$select))   cat("Select:   ", named_commas(x$select), "\n", sep = "")
}
