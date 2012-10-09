gmm_sample <- function(N) {
    y <- sample(c(1,-1), size=N, replace=TRUE, prob=c(.7,.3))
    x <- sapply(y, function(t) ifelse(t>0, rnorm(1,-10.,2.),rnorm(1,5.,3.)))
    matrix(c(x, y), 2, N, byrow=TRUE)
}

N <- 5
inst <- gmm_sample(N)
#print(instances)

png("test.png")
hist(inst[1,], breaks=100)

M <- 2
X <- inst[1,]
Y <- sample(c(1, -1))
Alpha   <- c(.5,.5)
Miu     <- runif(M)
Sigma   <- runif(M, 1, 10)
Gamma   <- matrix(rep(0., N*M), nrow = M)

#print(Gamma)

T <- 5
eps <- 1e-4
for (step in 1:T) {
    for (k in 1:M) {
        Gamma[k,] <- Alpha[k]*sapply(X, dnorm, Miu[k], Sigma[k])
        #print(Gamma[j,])
    }

    print(Gamma)
    tmp <- colSums(Gamma)
    Gamma <- Gamma/tmp
    #print(tmp)

    for (k in 1:M) {
        Nk          <- sum(Gamma[k,])
        Miu[k]      <- sum(Gamma[k,]*X)/Nk
        Sigma[k]    <- sum((X-Miu[k])%*%(X-Miu[k]))/Nk
        Alpha[k]    <- Nk/N
    }
}

print(Miu)
print(Sigma)
