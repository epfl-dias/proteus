# mapping between types in CSV and R types
type_map <- list(integer="integer(0)", varchar="character(0)", boolean="logical(0)")

# method used for mapping the types and handling some exceptions in direct type mapping
mapJDBCType <- function(JDBCType) {

  type <- tolower(JDBCType)

  if(grepl("varchar", type))
    type <- "varchar"

  return(type_map[[type]])
}
