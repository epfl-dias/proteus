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
setMethod("vidarDataType", signature("logical"),    function(x) "SMALLINT")
setMethod("vidarDataType", signature("Date"),       function(x) "DATE")
setMethod("vidarDataType", signature("difftime"),   function(x) "TIME")
setMethod("vidarDataType", signature("POSIXct"),    function(x) "TIMESTAMP")
setMethod("vidarDataType", signature("character"),  varchar_data_type)
setMethod("vidarDataType", signature("factor"),     varchar_data_type)
setMethod("vidarDataType", signature("list"),       list_data_type)
setMethod("vidarDataType", signature("AsIs"),       as_is_data_type)
