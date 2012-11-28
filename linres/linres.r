
linres_mle<-function(X, Y) {
    w<-solve(t(X)%*%X)%*%t(X)%*%matrix(Y,ncol=1)
    w
}

main<-function() {
    N<-100
    X<-matrix(c(runif(N,-5.,5.), rep(1., N)), ncol=2)
    Y<-sapply(X[,1], function(x) rnorm(1, .5*x+1., .5))

    w<-linres_mle(X, Y)

    D<-dim(X)[2]

    # demo it
    if (D==2) {
        png("image/2d_regres_mle.png")
        plot(X[,1], Y, xlim=c(-5.,5.), ylim=c(-5.,5.))

        lx<-c(-5.,5.)
        ly<-sapply(lx, function(x) w[1]*x+w[2])
        lines(lx, ly)
    }

    M<-10
    outliner<-sample(1:100, M)
    Y[outliner]<-sapply(outliner, function(i) rnorm(1, .5*X[i,1]+1., 8.))

    w<-linres_mle(X, Y)

    D<-dim(X)[2]
    # demo if

    if (D==2) {
        png("image/2d_regres_mle_outliner.png")
        plot(X[,1], Y, xlim=c(-5.,5.), ylim=c(-5.,5.))

        lx<-c(-5.,5.)
        ly<-sapply(lx, function(x) w[1]*x+w[2])
        lines(lx, ly)
    }
}

main()
