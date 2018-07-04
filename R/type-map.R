# mapping between types in CSV and R types
type_map <- list(integer="integer(0)", int="integer(0)", smallint="integer(0)",
                 varchar="character(0)", string="character(0)",
                 boolean="logical(0)",
                 double="double(0)", dbl="double(0)")

# method used for mapping the types and handling some exceptions in direct type mapping
mapJDBCType <- function(JDBCType) {

  type <- trimws(tolower(JDBCType))

  if(grepl("^varchar", type))
    type <- "varchar"

  return(type_map[[type]])
}
