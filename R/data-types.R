# mapping between types SQL->R, received from JDBC
type_map <- list(integer="integer(0)", int="integer(0)", smallint="integer(0)",
                 varchar="character(0)", string="character(0)", character="character(0)",
                 boolean="logical(0)",
                 double="double(0)", dbl="double(0)", numeric="double(0)")

# method used for mapping the types and handling some exceptions in direct type mapping
mapJDBCType <- function(JDBCType) {

  type <- trimws(tolower(JDBCType))

  ret <- c()

  for(el in type) {
    if(grepl("^varchar", el))
      el <- "varchar"

    ret <- c(ret, type_map[[el]])
  }

  return(ret)
}

# Setting up the R->SQL type mapping
setGeneric("vidarDataType",
           def = function(x, ...) standardGeneric("vidarDataType")
)

data_frame_data_type <- function(x) {
  vapply(x, vidarDataType, FUN.VALUE = character(1), USE.NAMES = TRUE)
}

varchar_data_type <- function(x) {
  "VARCHAR"
}

list_data_type <- function(x) {
  check_raw_list(x)
  "BLOB"
}

check_raw_list <- function(x) {
  is_raw <- vapply(x, is.raw, logical(1))
  if (!all(is_raw)) {
    stop("Only lists of raw vectors are currently supported", call. = FALSE)
  }
}

as_is_data_type <- function(x) {
  oldClass(x) <- oldClass(x)[-1]
  vidarDataType(x)
}

setOldClass("difftime")
setOldClass("AsIs")

setMethod("vidarDataType", signature("data.frame"), data_frame_data_type)
setMethod("vidarDataType", signature("integer"),    function(x) "INT")
setMethod("vidarDataType", signature("numeric"),    function(x) "DOUBLE")
#setMethod("vidarDataType", signature("double"),     function(x) "DOUBLE")
setMethod("vidarDataType", signature("logical"),    function(x) "BOOLEAN")
setMethod("vidarDataType", signature("Date"),       function(x) "DATE")
setMethod("vidarDataType", signature("difftime"),   function(x) "TIME")
setMethod("vidarDataType", signature("POSIXct"),    function(x) "TIMESTAMP")
setMethod("vidarDataType", signature("character"),  varchar_data_type)
setMethod("vidarDataType", signature("factor"),     varchar_data_type)
setMethod("vidarDataType", signature("list"),       list_data_type)
setMethod("vidarDataType", signature("AsIs"),       as_is_data_type)
