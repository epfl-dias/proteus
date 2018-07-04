#' @export
sql_select.ViDaRConnection <- function(con, select, from, where = NULL,
                                       group_by = NULL, having = NULL,
                                       order_by = NULL,
                                       limit = NULL,
                                       distinct = FALSE,
                                       ...) {

  out <- vector("list", 7)
  names(out) <- c("select", "from", "where", "group_by", "having", "order_by", "limit")

  out$select <- dbplyr:::sql_clause_select(select, con, distinct)

  # We use the connection environment to carry the information about unnest from for_all function
  # Try catch in certain cases (e.g. test connections) is required
  tryCatch(expr = {
    if(is.null(con@env$unnest)){
      # Unnest is not requested (NULL) - regular from clause
      out$from <- dbplyr:::sql_clause_from(from, con)
    } else {
      # Otherwise modify the regular from clause with unnest part

      out$from <- build_sql(sql("FROM"), " ", escape(from, collapse = ", ", con = con), ", ",
                            sql("UNNEST("), escape(con@env$unnest, collapse = ", ", con = con), ")")

      # Dealocate the unnest from the environment in case of nested queries -
      # we need to see them only the first time they appear
      con@env$unnest <- NULL
    }},
    # in case of test connection which does not have @env
    error = function(cond){
      out$from <<- dbplyr:::sql_clause_from(from, con)
    }
  )
  out$where <- dbplyr:::sql_clause_where(where, con)

  # TO CHECK case for not querying when the condition is (0 = 1) - lazy load
  #if(where == translate_sql(0L == 1L))
  #  return(escape(build_sql(sql("LAZY "), from)))

  out$group_by <- dbplyr:::sql_clause_group_by(group_by, con)
  out$having <- dbplyr:::sql_clause_having(having, con)
  out$order_by <- dbplyr:::sql_clause_order_by(order_by, con)

  # Limits are not supported (yet)
  #out$limit <- dbplyr:::sql_clause_limit(limit, con)

  dbplyr::escape(unname(dbplyr:::compact(out)), collapse = " ", parens = FALSE, con = con)
}

# Future use - checking for presence of certain vector functions in parts of queries
check_operators <- function(clause, operator) {
  for(el in clause){
    if(grepl(operator, el)){
      # If function is present, perform certain operations or query modifications
      silent(TRUE)
    }
  }
}

# Set the environment and supported functions (vector) by ViDaR
sql_translate_env.ViDaRConnection <- function(con) {
  sql_variant(
    sql_translator(.parent = base_scalar,
                   count = sql_prefix("count"),
                   collect = sql_prefix("collect"),
                   slice_index = function(what, ind) {
                     build_sql(sql(deparse(substitute(what))), sql("["), ind, sql("]"))
                   }
    ),
    sql_translator(.parent = base_agg
                   #collect = sql_prefix("collect")
                   # e.g. for_all = sql_aggregate_2("unnest")
    ),
    sql_translator(.parent = base_no_win
    )
  )
}
