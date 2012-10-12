gmm_sample <- function(N) {
    y <- sample(c(1,-1), size=N, replace=TRUE, prob=c(.7,.3))
    x <- sapply(y, function(t) ifelse(t>0, rnorm(1,-10.,2.),rnorm(1,5.,3.)))
    matrix(c(x, y), 2, N, byrow=TRUE)
}

gmm <- function(X, Iter) {
    N <- length(X)
    M <- 2
    Alpha   <- c(.5,.5)
    Miu     <- runif(M)
    Sigma   <- runif(M,1.,10.)
    Gamma   <- matrix(rep(0., N*M), nrow = M)

    eps <- 1e-4
    for (step in 1:Iter) {
        for (k in 1:M) {
            Gamma[k,] <- Alpha[k]*sapply(X, dnorm, Miu[k], Sigma[k])
        }

        tmp <- colSums(Gamma)
        for (k in 1:M) {
            Gamma[k,] <- Gamma[k,]/tmp
        }

        for (k in 1:M) {
            Nk          <- sum(Gamma[k,])
            Miu[k]      <- sum(Gamma[k,]*X)/Nk
            Sigma[k]    <- sqrt(sum(Gamma[k,]*(X-Miu[k])^2)/Nk)
            Alpha[k]    <- Nk/N
        }

        cat('step',step,'miu',Miu,'sigma',Sigma,'alpha',Alpha,'\n')
    }
}

main <- function() {
    N <- 10000
    inst <- gmm_sample(N)
    #print(instances)

    png("plot.png")
    hist(inst[1,], breaks=100)

    gmm(inst[1,],30)
}

main()
