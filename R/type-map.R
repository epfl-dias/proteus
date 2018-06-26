# mapping between types in CSV and R types
type_map <- list(integer="integer(0)", varchar="character(0)", boolean="logical(0)")

# method used for mapping the types and handling some exceptions in direct type mapping
mapJDBCType <- function(JDBCType) {

  type <- trimws(tolower(JDBCType))

  if(grepl("^varchar", type))
    type <- "varchar"

  return(type_map[[type]])
}

# method for parsing complex record types strings
parseRecordType <- function(parent, string){
  str_build <- c()

  string <- gsub("^RecordType\\(", "", string) %>% gsub("\\)[^\\)]*$", "", .) # remove the leading 'RecordType(' and trailing ')'

  str_locate_all(string, "RecordType\\(")

  complex_loc <- str_locate_all(string, "RecordType\\([^(\\)\\())]*\\)")  # get all positions of RecordType(...) in the string


  tokens <- strsplit(string, ",")[[1]]

  for(el in tokens) {
    el_tokens <- strsplit(el, " ")[[1]]

    type <- paste(head(el_tokens, n=-1), collapse = " ")
    name <- tail(el_tokens, n=1)

    str_build <- c(str_build, paste0(paste0(parent, ".", name),"=",mapJDBCType(type)))
  }

  return(str_build)
}

nesting_keywords <- c("recordtype")

# pairs of corresponding opening and closing brackets in the string
findNestings <- function(string, keyword = "RecordType") {
  openings <- str_locate_all(string, keyword)
  closings <- str_locate_all(string, "\\)")

  open_pos <- openings[[1]][,1]
  close_pos <- closings[[1]][,1]

  comparing <- expand.grid(open_pos, close_pos)

  distance <- comparing$Var2 - comparing$Var1

  dim = length(open_pos)

  res <- matrix(distance, nrow=dim, ncol=dim)
  res[res<0] = max(res)+1 # make it so it does not satisfy the condition of minimum, since we seek only non-negative distances

  pairs <- c()
  res <- a

  while(!is.null(dim(res)[1])){
    pair <- findMin(res)

    pairs <- c(pairs, c(open_pos[pair[1]], close_pos[pair[2]]))
    res <- res[-pair[1],-pair[2]]

    open_pos <- open_pos[-pair[1]]
    close_pos <- close_pos[-pair[2]]
  }

  pairs <- c(pairs, c(open_pos, close_pos))

  return(pairs)
}

findMin <- function(matrix) {
  which(matrix==min(matrix), arr.ind = TRUE)
}

# BETTER USE THIS FUNCTION FOR DISCOVERING PARENTHESES PAIRS
# AFTERWARDS SUBSTRING WITH GIVEN DELIMITERS FOR CUSTOM PARSING
findPairs <- function(string) {
  opening <- c()
  closing <- c()
  pairs <- c()
  pos <- 0

  for(char in strsplit(string, "")[[1]]){
    pos <- pos + 1
    if(char=='('){
      opening <- c(opening, pos)
    }
    else if(char==')'){
      closing <- c(closing, pos)

      pairs <- c(pairs, c(tail(opening, n=1), tail(closing, n=1)))

      opening <- head(opening, n=-1)
      closing <- head(closing, n=-1)
    }
  }

  return(pairs)
}

# string <- "RecordType(VARCHAR CHARACTER SET \"ISO-8859-1\" COLLATE \"ISO-8859-1$en_US$primary\" name2, INTEGER age2, RecordType(VARCHAR A, VARCHAR B, RecordType(VARCHAR A) ch1) ch1, RecordType(VARCHAR A, VARCHAR B) ch2) NOT NULL ARRAY"
