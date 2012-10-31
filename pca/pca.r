data<-function() {
    X<-matrix(c(2.5, 2.4,
        0.5, 0.7,
        2.2, 2.9,
        1.9, 2.2,
        3.1, 3.0,
        2.3, 2.7,
        2.,  1.6,
        1.,  1.1,
        1.5, 1.6,
        1.1, 0.9), ncol=2, byrow=T)
}

main<-function() {
    X<-data()
    N<-length(X[,1])

    png("test.png")
    plot(X[,1], X[,2], xlim=c(0.,3.5),ylim=c(0.,3.5))

    XAdj<-t(t(X)-colMeans(X))

    S<-cov(XAdj)
    lam<-eigen(S)$values
    vec<-eigen(S)$vectors

    print(vec)

    proj<-vec[, order(lam, decreasing=T)[1]]

    Xp<-X%*%proj
    print(Xp)
}

main()
