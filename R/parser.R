#string <- "RecordType(VARCHAR CHARACTER SET \"ISO-8859-1\" COLLATE \"ISO-8859-1$en_US$primary\" name2, INTEGER age2, RecordType(VARCHAR A, VARCHAR B, RecordType(VARCHAR A) ch1) ch1, RecordType(VARCHAR A, VARCHAR B) ch2) NOT NULL ARRAY"

processRecord <- function(string) {
  # remove everything until first parens from left and right
  s <- tolower(string)
  s <- gsub("^([^\\()]+\\()","\\(", s)
  s <- gsub("(\\)[^\\)]+)$","\\)", s)

  return(s)
}

tokenize <- function(string) {
  s <- gsub("([a-zA-Z0-9\"$\\._-]+)"," \\1 ", string)
  s <- gsub("\\s+", " ", s)

  return(strsplit(s, " "))
}

nestings <- c("recordtype")

parseRecordType <- function(string, parent, verbose = FALSE){

  output <- c()
  lookahead <- c()
  names <- c()
  types <- c()
  lookaheadRecord <- FALSE
  getName <- FALSE
  matchingParens <- 0

  s <- processRecord(string)
  tokens <- tokenize(s)

  for(token in tokens[[1]]){

    if(token %in% nestings){
      # look ahead the longest matching parens and call recursively
      lookaheadRecord <- TRUE
      lookahead <- c(lookahead, token)
      next
    }

    if(token == ","){
      # in this case, take the last one as name, and other as type
      if(!lookaheadRecord) {
        if(length(lookahead)>0){
          types <- c(types, paste(lookahead[1:length(lookahead)-1], collapse=" "))
          names <- c(names, paste0(parent,".",lookahead[length(lookahead)]))

          if(verbose){
            print("types")
            print(types)
            print("names")
            print(names)
          }

          lookahead <- c()
        }
      } else {
        lookahead <- c(lookahead, token)
      }
      next
    }

    if(token == "(") {
      if(lookaheadRecord){
        lookahead <- c(lookahead, token)
        matchingParens <- matchingParens+1
      }
      next
    }

    if(token == ")") {
      if(lookaheadRecord){
        lookahead <- c(lookahead, token)
        matchingParens <- matchingParens-1

        # longest match found, read one more token that is the name of the structure
        if(matchingParens==0){
          lookaheadRecord <- FALSE
          getName <- TRUE
        }

      } else {
        if(length(lookahead)>0){
          types <- c(types, paste(lookahead[1:length(lookahead)-1], collapse=" "))
          names <- c(names, paste0(parent,".",lookahead[length(lookahead)]))

          if(verbose) {
            print("types")
            print(types)
            print("names")
            print(names)
          }

          lookahead <- c()
        }
      }
      next
    }

    # regular case
    lookahead <- c(lookahead, token)

    if(getName){
      getName <- FALSE
      # last token in the name of the structure, and pass it to function call
      input <- paste(lookahead[1:length(lookahead)-1], collapse = " ")
      name <- paste0(parent, ".", lookahead[length(lookahead)])

      lookahead <- c()

      if(verbose){
        print("RECURSIVELY:")
        print(input)
      }

      output <- c(output, parseRecordType(input, name))

    }
  }

  output <- c(output,  paste0(names, "=", mapJDBCType(types)))

  return(output)
}

