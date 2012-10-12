parzen_sample <- function(N) {
    rnorm(N,5.,3.)
}

parzen <- function(X, func="cube", Vn="sqrt") {
    N <- length(X)
    A <- min(X)
    B <- max(X)

    if (Vn == "sqrt") { V<-1./sqrt(N) }
    else { V<-1./log(N)}

    S <- seq(A,B,.01)
    if (func == "cube") { Y<-lapply(S, function(x) {length(X[X>x-.5*V & X<x+.5*V])/V/N}) }
    else { Y<-lapply(S, function(x) {sum(dnorm(x - X))/V/N}) }

    Y<-lapply(S, function(x) {length(X[X>x-.5*V & X<x+.5*V])/V/N})

    matrix(c(S,Y),nrow=2,byrow=T)
}

main <- function() {
    N <- 10000
    X <- parzen_sample(N)
    png("parzen_1d_demo_hist.png")
    hist(X, breaks=100)

    # set window function cubic
    G<-parzen(X,func="cube",Vn="sqrt")
    png("parzen_1d_demo_1.png")
    plot(G[1,],G[2,],type="n")
    lines(G[1,],G[2,])

    G<-parzen(X,func="cube",Vn="ln")
    png("parzen_1d_demo_2.png")
    plot(G[1,],G[2,],type="n")
    lines(G[1,],G[2,])

    G<-parzen(X,func="gauss",Vn="sqrt")
    png("parzen_1d_demo_3.png")
    plot(G[1,],G[2,],type="n")
    lines(G[1,],G[2,])

    G<-parzen(X,func="gauss",Vn="ln")
    png("parzen_1d_demo_4.png")
    plot(G[1,],G[2,],type="n")
    lines(G[1,],G[2,])
}

main()
