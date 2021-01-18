for_each <- function(.data, ...) {
  dots <- as.character(quos(...))
  add_op_single("for_each", .data, dots = dots)
}


sql_clause_unnest <- function(from, unnest, con){

  unnest <- gsub("~", "", as.character(unnest))

  if (length(from) > 0L && length(unnest) > 0L) {
    assertthat::assert_that(is.character(from))
    assertthat::assert_that(is.character(unnest))

    from_name <- random_table_name()

    build_sql(
      "FROM ",
      sql(paste(from, from_name)),
      ", ",
      sql(paste0("UNNEST (", ident(from_name), ".", ident(unnest), ") AS ", ident(unnest), collapse = ", ")),
      con = con
    )
  }
}


sql_unnest <- function(con, select, from, unnest, ...) {

  out <- vector("list", 3)
  names(out) <- c("select", "from", "unnest")

  if(is.ident(from)){
    con@env$last_from <- from
  }

  out$select <- dbplyr:::sql_clause_select(select, con, distinct = FALSE)

  out$from <- sql_clause_unnest(from, unnest, con)

  escape(unname(purrr::compact(out)), collapse = " ", parens = FALSE, con = con)
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
    con, query$select, from, query$unnest,
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
