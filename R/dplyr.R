# definitions or redefinitions of dplyr verbs for ViDaR
library(dbplyr)
library(dplyr)

slice_sql <- function(what, ind)
{
  build_sql(sql(deparse(substitute(what))), sql("["), ind, sql("]"))
}

# vidar aggregate functions
vidar_agg <- sql_translator(.parent = base_agg
  #for_all = sql_aggregate_2("unnest")
)

# vidar window functions
vidar_win <- sql_translator(.parent = base_no_win
)

# vidar scalar functions
vidar_scalar <- sql_translator(.parent = base_scalar,
  count = sql_prefix("cardinality"),
  get_all = sql_prefix("collect"),
  slice_index = slice_sql
)

# wrapping it up
vidar_var <- sql_variant(
  vidar_scalar,
  vidar_agg,
  vidar_win
)

# set the environment
sql_translate_env.ViDaRConnection <- function(x) vidar_var

# custom build sql for indexing


# ... %>% sql_build(.)
# ... ident(...)
# sql_render
# show_query


# unnest try
emp_jsn = '{"name":"string", "age":"int", "children":[{"name2":"string", "age2":"int"}]}'
df_emp <- data.frame(jsonlite::fromJSON(emp_jsn, flatten = TRUE, simplifyDataFrame = TRUE))
emp <- as.tbl(df_emp)


emp <- tbl(con, "emp")
