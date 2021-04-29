sql.stddev <- function(tbl_values) {
  fields <- sapply(colnames(tbl_values), function(x){ paste0("stdev.", x, " AS stddev(", x, ")") })
  cmd <- paste0("SELECT ", paste0(fields, collapse = ", "), " FROM ", tbl_values[["ops"]]$x)

  #return(eval(parse(text=cmd)))
  return(cmd)
}


# at a later step, initialization of the parameters from data: lambda, mean and standard deviation,
# all based on splitting the data in K bins and finding the distribution parameters on the bins
# for now - we presume the values are given, not sure how to efficiently do this in ViDa (sorting data, then splitting in k ~bins and then finding stddev and mean)
# ONE SOLUTION:
# survey the data several times (e.g. first presumption that the data is uniform, and then adapt in several iters) depending on the count of the bins
# in this case IF clause with COUNT should be used to determine if the bins are of comparable size (and just later apply sort when bin size is found)
# for case 2 it is easy - simply use the mean value, maybe even for even numbers (e.g. binning and then repeating with mean over sub-bin)
# mixtools init
normalmix.init=function (x, lambda = NULL, mu = NULL, s = NULL, k = 2, arbmean = TRUE,
                         arbvar = TRUE)
{
  if (!is.null(s)) {
    arbvar <- (length(s) > 1)
    if (arbvar)
      k <- length(s)
  }
  if (!is.null(mu)) {
    arbmean <- (length(mu) > 1)
    if (arbmean) {
      k <- length(mu)
      if (!is.null(s) && length(s) > 1 && k != length(s)) {
        stop("mu and sigma are each of length >1 but not of the same length.")
      }
    }
  }
  if (!arbmean && !arbvar) {
    stop("arbmean and arbvar cannot both be FALSE")
  }
  n = length(x)
  x = sort(x)
  x.bin = list()
  for (j in 1:k) {
    x.bin[[j]] <- x[max(1, floor((j - 1) * n/k)):ceiling(j *
                                                           n/k)]
  }
  if (is.null(s)) {
    s.hyp = as.vector(sapply(x.bin, sd))
    if(any(s.hyp==0)) s.hyp[which(s.hyp==0)] = runif(sum(s.hyp==0),0,sd(x))
    if (arbvar) {
      s = 1/rexp(k, rate = s.hyp)
    }
    else {
      s = 1/rexp(1, rate = mean(s.hyp))
    }
  }
  if (is.null(mu)) {
    mu.hyp <- as.vector(sapply(x.bin, mean))
    if (arbmean) {
      mu = rnorm(k, mean = mu.hyp, sd = s)
    }
    else {
      mu = rnorm(1, mean = mean(mu.hyp), sd = mean(s))
    }
  }
  if (is.null(lambda)) {
    lambda <- runif(k)
    lambda <- lambda/sum(lambda)
  }
  else {
    lambda <- rep(lambda, length.out = k)
    lambda <- lambda/sum(lambda)
  }
  list(lambda = lambda, mu = mu, s = s, k = k, arbvar = arbvar,
       arbmean = arbmean)
}

# interface similar to the  one from mixtools (to preserve the familiarity and as a baseline for testing)
normalmixEM <- function (x, lambda = NULL, mu = NULL, sigma = NULL, k = 2,
            mean.constr = NULL, sd.constr = NULL,
            epsilon = 1e-08, maxit = 1000, maxrestarts=20,
            verb = FALSE, fast=FALSE, ECM = FALSE,
            arbmean = TRUE, arbvar = TRUE) {


    x <- as.vector(x)

    # setting up variables
    tmp <- normalmix.init(x = x, lambda = lambda, mu = mu, s = sigma,
                          k = k, arbmean = arbmean, arbvar = arbvar)



    lambda <- tmp$lambda
    mu <- tmp$mu
    sigma <- tmp$s
    k <- tmp$k
    arbvar <- tmp$arbvar
    arbmean <- tmp$arbmean


      z <- parse.constraints(mean.constr, k=k, allsame=!arbmean)
      meancat <- z$category; meanalpha <- z$alpha
      z <- parse.constraints(sd.constr, k=k, allsame=!arbvar)
      sdcat <- z$category; sdalpha <- z$alpha
      ECM <- ECM || any(meancat != 1:k) || any(sdcat != 1)
      n <- length(x)
      notdone <- TRUE
      restarts <- 0
      while(notdone) {
        # Initialize everything
        notdone <- FALSE
        tmp <- normalmix.init(x = x, lambda = lambda, mu = mu, s = sigma,
                              k = k, arbmean = arbmean, arbvar = arbvar)
        lambda <- tmp$lambda
        mu <- tmp$mu
        k <- tmp$k
        sigma <- tmp$s
        var <- sigma^2
        diff <- epsilon+1
        iter <- 0
        postprobs <- matrix(nrow = n, ncol = k)
        mu <- rep(mu, k)[1:k]
        sigma <- rep(sigma,k)[1:k]
        # Initialization E-step here:
        z <- .C(C_normpost, as.integer(n), as.integer(k),
                as.double(x), as.double(mu),
                as.double(sigma), as.double(lambda),
                res2 = double(n*k), double(3*k), post = double(n*k),
                loglik = double(1), PACKAGE = "mixtools")
        postprobs <- matrix(z$post, nrow=n)
        res <- matrix(z$res2, nrow=n)
        ll <- obsloglik <- z$loglik
        while (diff > epsilon && iter < maxit) {
          # ECM loop, 1st M-step: condition on sigma, update lambda and mu
          lambda <- colMeans(postprobs)
          mu[meancat==0] <- meanalpha[meancat==0]
          if (max(meancat)>0) {
            for(i in 1:max(meancat)) {
              w <- which(meancat==i)
              if (length(w)==1) {
                mu[w] <- sum(postprobs[,w]*x) / (n*lambda[w])
              } else {
                tmp <- t(postprobs[,w])*(meanalpha[w]/sigma[w]^2)
                mu[w] <- meanalpha[w] * sum(t(tmp)*x) / sum(tmp*meanalpha[w])
              }
            }
          }

          if (ECM) {  # If ECM==FALSE, then this is a true EM algorithm and
            # so we omit the E-step between the mu and sigma updates
            # E-step number one:
            z <- .C(C_normpost, as.integer(n), as.integer(k),
                    as.double(x), as.double(mu),
                    as.double(sigma), as.double(lambda),
                    res2 = double(n*k), double(3*k), post = double(n*k),
                    loglik = double(1), PACKAGE = "mixtools")
            postprobs <- matrix(z$post, nrow=n)
            res <- matrix(z$res2, nrow=n)

            # ECM loop, 2nd M-step: condition on mu, update lambda and sigma
            lambda <- colMeans(postprobs) # Redundant if ECM==FALSE
          }
          sigma[sdcat==0] <- sdalpha[sdcat==0]
          if (max(sdcat)>0) {
            for(i in 1:max(sdcat)) {
              w <- which(sdcat==i)
              if (length(w)==1) {
                sigma[w] <- sqrt(sum(postprobs[,w]*res[,w]) / (n*lambda[w]))
              } else {
                tmp <- t(postprobs[,w]) / sdalpha[w]
                sigma[w] <- sdalpha[w] * sqrt(sum(t(tmp) * res[,w])/ (n * sum(lambda[w])))
              }
            }
            if(any(sigma < 1e-08)) {
              notdone <- TRUE
              cat("One of the variances is going to zero; ",
                  "trying new starting values.\n")
              restarts <- restarts + 1
              lambda <- mu <- sigma <- NULL
              if(restarts>maxrestarts) { stop("Too many tries!") }
              break
            }
          }

          # E-step number two:
          z <- .C(C_normpost, as.integer(n), as.integer(k),
                  as.double(x), as.double(mu),
                  as.double(sigma), as.double(lambda),
                  res2 = double(n*k), double(3*k), post = double(n*k),
                  loglik = double(1), PACKAGE = "mixtools")
          postprobs <- matrix(z$post, nrow=n)
          res <- matrix(z$res2, nrow=n)
          newobsloglik <- z$loglik
          diff <- newobsloglik - obsloglik
          obsloglik <- newobsloglik
          ll <- c(ll, obsloglik)
          iter <- iter + 1
          if (verb) {
            cat("iteration =", iter, " log-lik diff =", diff, " log-lik =",
                obsloglik, "\n")
            print(rbind(lambda, mu, sigma))
          }
        }
      }
      if (iter == maxit) {
        cat("WARNING! NOT CONVERGENT!", "\n")
      }
      cat("number of iterations=", iter, "\n")
      if(arbmean == FALSE){
        scale.order = order(sigma)
        sigma.min = min(sigma)
        postprobs = postprobs[,scale.order]
        colnames(postprobs) <- c(paste("comp", ".", 1:k, sep = ""))
        a=list(x=x, lambda = lambda[scale.order], mu = mu, sigma = sigma.min,
               scale = sigma[scale.order]/sigma.min, loglik = obsloglik,
               posterior = postprobs, all.loglik=ll, restarts=restarts,
               ft="normalmixEM")
      } else {
        colnames(postprobs) <- c(paste("comp", ".", 1:k, sep = ""))
        a=list(x=x, lambda = lambda, mu = mu, sigma = sigma, loglik = obsloglik,
               posterior = postprobs, all.loglik=ll, restarts=restarts,
               ft="normalmixEM")
      }


    class(a) = "mixEM"
    options(warn) # Reset warnings to original value
    a
  }
