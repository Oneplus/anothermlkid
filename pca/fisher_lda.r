data<-function() {
    X<-matrix(c(1.,2.,
        2.,1.,
        2.,3.,
        3.,2.,
        6.,7.,
        7.,6.,
        7.,8.,
        8.,7.), ncol=2, byrow=T)
    Y<-c(1.,-1.,1.,-1.,1.,-1.,1.,-1)

    list(X=X,Y=Y)
}

main<-function() {
    D<-data()

    # plotting data
    png("image/fisher_lda.png")
    plot(D$X[which(D$Y==1),1], D$X[which(D$Y==1),2], xlim=c(0.,9.), ylim=c(0.,9.),pch=3)
    points(D$X[which(D$Y==-1),1], D$X[which(D$Y==-1),2],pch=4)

    miu<-rbind(colMeans(D$X[which(D$Y==1),]),colMeans(D$X[which(D$Y==-1),]))

    Xbar<-colMeans(D$X)

    Sb<-matrix(rep(0.,2*2),nrow=2)
    Sw<-matrix(rep(0.,2*2),nrow=2)

    for (c in 1:2) {
        Sb<-Sb+(miu[c,]-Xbar)%*%t(miu[c,]-Xbar)
    }

    for (c in 1:2) {
        for (xi in D$X[which(D$Y==ifelse(c==1,1,-1)),]) {
            Sw<-Sw+(xi-miu[c,])%*%t(xi-miu[c,])
        }
    }
    print(Sb)
    print(Sw)

    S<-solve(Sw)*Sb

    lam<-eigen(S)$value
    vec<-eigen(S)$vectors

    proj<-vec[, order(lam, decreasing=T)[1]]
    Xp<-D$X%*%proj

    print(lam)
    print(vec)
    print(proj)
    lines(c(0.,9.), c(0.,9.)*proj[1]/(-proj[2])+c(9.,9.))
}

main()
