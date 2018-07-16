# Contains R source code for ML functions utilizing DBI and dplyr
# presumes that data is a tbl with jdbc connection object

# K-means - for now only with distance metric for numeric types #
# kmeans.tbl_lazy for compatability if necessary
# future - if it will be part of dplyr pipe, see how to incorporate
# WHY - output of kmeans is k centroids (see dplyr.R for more info)
# kmeans <- function(.data, k = 10, max.iter = 5, ...) {
#
#   dim <- ncol(data)
#   dots <- quos(k, dim, max.iter, ...)
#
#   add_op_single("kmeans", .data, dots = dots)
# }
#
# sql_build.op_kmeans <- function(op, con, ...) {
#
#   select_query(op$x, con, ...)
# }

k_means <- function(data, k=3, max.iter=5){

  # dimensionality is of the number of columns
  dim <- ncol(data)
  # assert that data is numeric
  # con <- data[[1]]$con

  centroids <- randomCentroids(data, k = k)

  iters <- 0
  oldCentroids <- NULL


  # for each centroid check which points belong (lowest distance)

  # e.g. for k=3
  # SELECT SUM(aaa.x) AS sum_x, COUNT(aaa.x) AS cnt_x, SUM(aaa.y) as sum_y, CNT(aaa.y) as cnt_y, (CASE
  #                                 WHEN (aaa.p1<aaa.p2) AND (aaa.p1<aaa.p3) THEN 1
  #                                 WHEN (aaa.p2<aaa.p1) AND (aaa.p2<aaa.p3) THEN 2
  #                                 ELSE 3 END) AS member
  # FROM(SELECT x, y, SQRT(POW(x+1,2)+POW(y+2,2)) AS P1, SQRT(POW(x+1,2)+POW(y-1,2)) AS P2,
  #      SQRT(POW(x-3,2)+POW(y+3,2)) AS P3 FROM points) aaa
  # GROUP BY member;

  # query generation depending on the conditions (e.g. parameter K)

  # TODO push this as max.iter nested queries (one step )
  while(iters<max.iter){ # add delta as loop termination as well
    oldCentroids <- centroids

    # generate query using old centroids
    centroids <- calculateCentroids(data, k, oldCentroids)

    iters <- iters + 1
  }

  return(centroids)
}

calculateCentroids <- function(data, k, oldCentroids) {
  # e.g. for k=3 and euclidean distance metric
  # SELECT SUM(aaa.x) AS sum_x, COUNT(aaa.x) AS cnt_x, SUM(aaa.y) as sum_y, CNT(aaa.y) as cnt_y, (CASE
  #                                 WHEN (aaa.p1<aaa.p2) AND (aaa.p1<aaa.p3) THEN 1
  #                                 WHEN (aaa.p2<aaa.p1) AND (aaa.p2<aaa.p3) THEN 2
  #                                 ELSE 3 END) AS member
  # FROM(distances_query) aaa
  # GROUP BY member;
  col_names <- colnames(data)
  dim <- ncol(data)

  tmp_table_name <- "aaa"

  distances_query <- generateDistances(data, k, oldCentroids, templateEuclidean)

  # select_clause <- paste0("SELECT ", paste0("SUM(", col_names,") AS sum_", col_names, ", COUNT(",col_names,") AS cnt_", col_names, collapse = ", "))
  select_clause <- paste0("SELECT ", paste0("AVG(", col_names,") AS avg_", col_names, collapse = ", "))

  case_clause <- generateCase(k, tmp_table_name)

  # result of the query has to be processed (to get the actual centroids)
  query <- paste0(select_clause, ", (", case_clause, ") AS `member` ", "FROM (", distances_query, ") ", tmp_table_name, " GROUP BY (", case_clause,")")

  result <- dbFetch(dbSendQuery(data[[1]]$con, query))

  # divide sum with count (sums - 1) because we have the member column as odd included
  #r <- result[seq(1, ncol(result)-1, 2)] / result[seq(2, ncol(result), 2)]

  # TODO
  # check if result has k rows, if not, generate some

  return(result)
}

generateCase <- function(k, table_name) {
  case_clause <- c()

  # TODO can i do (k-1)+(k-2)... or for if-else I need to do k*k - tested, should be ok
  for(i in 1:(k-1)) {
    tmp <- c()

    for(j in (i+1):k) {
      tmp <- c(tmp, paste0("(", table_name, ".P", i ,"<" , table_name, ".P", j, ")"))
    }

    case_clause <- c(case_clause, paste0("WHEN ", paste0(tmp, collapse = " AND ")," THEN ", as.character(i)))

  }

  return(paste0("CASE ", paste0(case_clause, collapse = " "), " ELSE ", as.character(k) ," END"))
}

# distances_query
# SELECT x, y, SQRT(POW(x+1,2)+POW(y+2,2)) AS P1, SQRT(POW(x+1,2)+POW(y-1,2)) AS P2,
#      SQRT(POW(x-3,2)+POW(y+3,2)) AS P3 FROM points
generateDistances <- function(data, k, oldCentroids, metricTemplate = templateEuclidean) {
  col_names <- colnames(data)
  dim <- ncol(data)

  # if oldCentroids are a dataframe, convert them to vector
  if(is.data.frame(oldCentroids)) {
    oldCentroids <- as.numeric(as.vector(t(oldCentroids[, 1:(ncol(oldCentroids)-1)])))
  }

  distances_query <- paste0("SELECT ", paste(col_names, collapse = ", "), ", ")

  generator <- c()

  for(i in 0:(k-1)) {

    generator <- c(generator, paste0("(", metricTemplate(col_names, oldCentroids[(1+i*dim):(dim+i*dim)]), ") AS P", (i+1)))

  }

  print("TABLE NAME:")
  print(as.character(data[[2]]$x))
  # TODO uncomment when the source is tbl
  distances_query <- paste0(distances_query, paste(generator, collapse = ", "), " FROM ", as.character(data[[2]]$x)) # as.character(data[[2]]$x) - internal table name

  return(distances_query[[1]])
}

templateEuclidean <- function(data_column_names, point_coordinates) {
  #generates the list which is equivalent to (SUM(POW(data-point, 2)))
  sign <- sapply(point_coordinates, function(x) if(x<0) "+" else "-")
  paste0("(",data_column_names, sign, abs(point_coordinates), ")*(",data_column_names, sign, abs(point_coordinates), ")", collapse = "+")
}

# Function for generating random centroids, bounded by ranges in data (in order to have a sense of scale of data)
randomCentroids <- function(data, k) {
  cmd <- "data %>% summarise("
  list <- c()

  for(name in colnames(data)) {
    #list <- c(list, paste0("max_",name,"=max(",name,", na.rm=TRUE), min_",name,"=min(",name,", na.rm=TRUE)"))
    list <- c(list, paste0("max_",name,"=max(",name,", na.rm=TRUE)"))
  }

  cmd <- paste0(cmd, paste0(list, collapse = ","), ")")

  bounds <- as.data.frame(eval(parse(text = cmd)))

  ncols <- ncol(bounds)
  # return k random centroids in the given bounds (with given dimensionalities), calculate them from the vector
  #max_list <- as.numeric(bounds[1, seq(1, ncols, 2)])
  #min_list <- as.numeric(bounds[1, seq(2, ncols, 2)])
  max_list <- as.numeric(bounds)
  min_list <- as.numeric(0.5*bounds)

  # returns a list of numbers (dimensionality * k), first dim numbers represent a single centroid, arrange in order of data columns
  #centroids <- runif(k*ncols/2, min_list, max_list)
  centroids <- runif(k*ncols, min_list, max_list)
  return(centroids)
}
