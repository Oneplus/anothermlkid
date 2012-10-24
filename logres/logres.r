sample_data<-function(N,means,sd) {
    matrix(c(rnorm(N,means,sd), rnorm(N,means,sd), rep(1.,N)), ncol=3, byrow=F)
}

lr_without_regularization<-function(X,Y) {
    w<-solve(t(X)%*%X)%*%t(X)%*%Y
    w
}

lr_with_regularization<-function(X,Y) {
    N<-length(X)
    D<-dim(X)[2]
    epsilon<-0.00001
    gamma<-1.
    max_iter<-1000
    C<-0.01

    Y<-matrix(Y,ncol=1)
    w<-rep(0.,D)

    sigmoid<-function(z) { 1./(1.+exp(z)) }
    vec_norm<-function(x) { sqrt(sum(x^2)) }

    for(iter in 1:max_iter) {
        w_old<-w

        w<-w_old+gamma*t(X)%*%sigmoid(-X%*%matrix(w_old,ncol=1)%*%t(Y))%*%Y-gamma*C*w_old

        cat("iteration: ", iter, "\tw: ", w, "\n")
        if(vec_norm(w-w_old)<epsilon) { break }
    }
    w
}

demo_without_regularization<-function() {
    N<-500
    X1<-sample_data(N,4.,1.5)
    X2<-sample_data(N,0.,1.5)
    png("image/demo_without_regularization.png")
    plot(X1[,1], X1[,2], pch=3, xlim=c(-3.,7.),ylim=c(-3.,7.),col="red")
    points(X2[,1], X2[,2], pch=4, col="blue")

    X<-rbind(X1,X2)
    Y<-c(rep(1.,500),rep(-1.,500))

    w<-lr_without_regularization(X, Y)
    print(w)
    range_x<-seq(-1.,5.,by=.1)
    boundary<-function(x) {(-w[3]-w[1]*x)/w[2]}

    lines(range_x, lapply(range_x, boundary))
}

demo_with_regularization<-function() {
    N<-500
    X1<-sample_data(N,4.,1.5)
    X2<-sample_data(N,0.,1.5)

    png("image/demo_with_regularization.png")
    plot(X1[,1], X1[,2], pch=3, xlim=c(-3.,7.),ylim=c(-3.,7.),col="red")
    points(X2[,1], X2[,2], pch=4, col="blue")

    X<-rbind(X1,X2)
    Y<-c(rep(1.,500),rep(-1.,500))

    w<-lr_with_regularization(X, Y)
    print(w)
    range_x<-seq(-1.,5.,by=.1)
    boundary<-function(x) {(-w[3]-w[1]*x)/w[2]}

    lines(range_x, lapply(range_x, boundary))
}

main<-function(){
    demo_without_regularization()
    demo_with_regularization()
}

main()
