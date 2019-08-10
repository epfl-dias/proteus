#' @export
sql_select.ViDaRConnection <- function(con, select, from, where = NULL,
                                       group_by = NULL, having = NULL,
                                       order_by = NULL,
                                       limit = NULL,
                                       distinct = FALSE,
                                       ...) {

  out <- vector("list", 7)
  names(out) <- c("select", "from", "where", "group_by", "having", "order_by", "limit")

  if(is.ident(from)){
    con@env$last_from <- from
  }

  from_chr <- as.character(con@env$last_from)



  if(!is.null(con@env$name_map)){
    if(!is.null(con@env$name_map[[from_chr]])) {

      # new_select <- c()

      # for(el in select){
      #   if(!is.null(con@env$name_map[[from_chr]][[el]]))
      #     new_select <- c(new_select, con@env$name_map[[from_chr]][[el]])
      #   else
      #     new_select <- c(new_select, el)
      # }

      # select <- new_select
    }
  }


  out$select <- dbplyr:::sql_clause_select(select, con, distinct)

  out$from <- dbplyr:::sql_clause_from(from, con)

  out$where <- dbplyr:::sql_clause_where(where, con)

  # TO CHECK case for not querying when the condition is (0 = 1) - lazy load
  #if(where == translate_sql(0L == 1L))
  #  return(escape(build_sql(sql("LAZY "), from)))

  out$group_by <- dbplyr:::sql_clause_group_by(group_by, con)
  out$having <- dbplyr:::sql_clause_having(having, con)
  out$order_by <- dbplyr:::sql_clause_order_by(order_by, con)

  # Limits are not supported (yet)
  #out$limit <- dbplyr:::sql_clause_limit(limit, con)

  escape(unname(purrr::compact(out)), collapse = " ", parens = FALSE, con = con)
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
                   }#,
                   #kmeans1 = function(data, k=10, max.iter=5) {
                  #  build_sql(sql("KMEANS") , sql("k = "), sql(k))
    ),
    sql_translator(.parent = base_agg
                   #collect = sql_prefix("collect")
                   #for_all = sql_aggregate_2("unnest")
    ),
    sql_translator(.parent = base_no_win
    )
  )
}
