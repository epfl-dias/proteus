# for_all - dplyr definition of function acting as unnest alias
for_all <- function(.data, ...) {
  dots <- quos(...)
  add_op_single("for_all", .data, dots = dots)
}

# sql_build definition of a function - definition of behavior of dbplyr function
sql_build.op_for_all <- function(op, con, ...) {

  # assign the unnest fields to the connection environment
  if(!is.null(con))
    con@env$unnest <- get_unnests(op$dots)

  # pass the operations further on without any modifications
  return(sql_build(op$x, con, ...))
}

# Utility function for extracting the fields to unnest as idents (as required for the from clause)
get_unnests <- function(quosures) {
  unnests <- c()
  for(q in quosures){
    unnests <- c(unnests, (as.character(get_expr(q))))
  }

  return(ident(unnests))
}

