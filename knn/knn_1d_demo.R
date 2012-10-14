bisect<-function(X, x) {
    low<-1
    high<-length(X)

    while (low+1<high) {
        mid<-floor((low+high)/2)
        if (x<X[mid]) { high<-mid }
        else { low<-mid }
    }

    mid
}

knn_sample<-function(N) {
    rnorm(N,5.,3.)
}

knn_estimate<-function(X) {
    X<-sort(X)
    N<-length(X)
    A<-X[1]
    B<-X[N]
    K<-sqrt(N)

    S<-seq(A,B,.01)

    Prob<-function(x) {
        i<-bisect(X,x)
        k<-0
        L<-i - 1
        R<-i + 1
        while (k<K) {
            if (L<1) { R<-R+1 }
            else if (R>N) { L<-L-1 }
            else {
                if (abs(X[L]-x)>abs(X[R]-x)) { R<-R+1 }
                else { L<-L-1 }
            }

            k<-k+1
        }
        L<-ifelse(L<1,1,L)
        R<-ifelse(R>N,N,R)
        K/(X[R]-X[L])/N
    }

    Y<-lapply(S, Prob)

    matrix(c(S,Y),nrow=2,byrow=T)
}

main<-function() {
    N<-10000
    X<-knn_sample(N)
    png("knn_1d_demo_hist.png")
    hist(X, breaks=100)
    G<-knn_estimate(X)

    png("knn_1d_demo_1.png")
    plot(G[1,],G[2,],type="n")
    lines(G[1,],G[2,])
}

main()
