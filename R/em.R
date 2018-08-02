sql.stddev <- function(tbl_values) {
  fields <- sapply(colnames(tbl_values), function(x){ paste0("stdev.", x, "=stddev(", x, ")") })
  cmd <- paste0("tbl_values %>% summarise(",paste0(fields, collapse = ", "),")")

  return(eval(parse(text=cmd)))
}



# interface similar to the  one from mixtools (to preserve the familiarity and have a testing reference)
normalmixEM <- function (x, lambda = NULL, mu = NULL, sigma = NULL, k = 2,
            mean.constr = NULL, sd.constr = NULL,
            epsilon = 1e-08, maxit = 1000, maxrestarts=20,
            verb = FALSE, fast=FALSE, ECM = FALSE,
            arbmean = TRUE, arbvar = TRUE) {


  }
