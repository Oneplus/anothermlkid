sigmoid<-function(z) {
    1./(1.+exp(-z))
}

vec_norm<-function(x) {
    sqrt(sum(x^2))
}

sample_data<-function(N,means,sd) {
    matrix(c(rnorm(N,means,sd), rnorm(N,means,sd), rep(1.,N)), ncol=3, byrow=F)
}

lr_without_regularization<-function(X,Y) {
    w<-solve(t(X)%*%X)%*%t(X)%*%Y
    w
}

lr_with_regularization<-function(X,Y,C) {
    N<-length(X)
    D<-dim(X)[2]
    epsilon<-0.001
    gamma<-0.0005
    max_iter<-5000

    w<-rep(0.,D)

    for(iter in 1:max_iter) {
        w_old<-w

        w<-w_old+gamma*t(Y*X)%*%(sigmoid(-Y*X%*%matrix(w_old,ncol=1)))-gamma*C*w_old

        cat("iteration: ", iter, "\tw: ", w, "\n")
        if(vec_norm(w-w_old)<epsilon) { break }
    }
    w
}

demo_without_regularization<-function() {
    N<-500
    box<-c(-3.,7.)
    X1<-sample_data(N,4.,1.5)
    X2<-sample_data(N,0.,1.5)
    png("image/demo_without_regularization.png")
    plot(X1[,1], X1[,2], pch=3, xlim=box,ylim=box,col="red")
    points(X2[,1], X2[,2], pch=4, col="blue")

    X<-rbind(X1,X2)
    Y<-c(rep(1.,N),rep(-1.,N))

    w<-lr_without_regularization(X, Y)
    range_x<-seq(-1.,5.,by=.1)
    boundary<-function(x) {(-w[3]-w[2]*x)/w[1]}

    lines(range_x, lapply(range_x, boundary))

    png("image/error_function_surface_without_regularization.png")
    Ew<-function(w1, w2) {
        -sum(log(sigmoid(Y*(w1*X[,1]+w2*X[,2]))))
    }

    plot_x<-seq(-10.,10.,length.out=50)
    plot_y<-seq(-10.,10.,length.out=50)
    plot_z<-outer(plot_x, plot_y, function(ai,bj) mapply(Ew,ai,bj))

    persp(plot_x, plot_y, plot_z)

}

demo_with_regularization<-function() {
    N<-500
    box<-c(-3.,7.)
    X1<-sample_data(N,4.,1.5)
    X2<-sample_data(N,0.,1.5)

    png("image/demo_with_regularization.png")
    plot(X1[,1], X1[,2], pch=3, xlim=box,ylim=box,col="red")
    points(X2[,1], X2[,2], pch=4, col="blue")

    X<-rbind(X1,X2)
    Y<-c(rep(1.,N),rep(-1.,N))

    C<-2.
    w<-lr_with_regularization(X, Y, C)
    range_x<-seq(-1.,5.,by=.1)
    boundary<-function(x) {(-w[3]-w[1]*x)/w[2]}

    lines(range_x, lapply(range_x, boundary))

    png("image/error_function_surface_with_regularization.png")
    Ew<-function(w1, w2) {
        -sum(log(sigmoid(Y*(w1*X[,1]+w2*X[,2]))))+C*0.5*vec_norm(c(w1,w2))
    }

    plot_x<-seq(-10.,10.,length.out=50)
    plot_y<-seq(-10.,10.,length.out=50)
    plot_z<-outer(plot_x, plot_y, function(ai,bj) mapply(Ew,ai,bj))

    persp(plot_x, plot_y, plot_z)
} 

main<-function(){
    demo_without_regularization()
    demo_with_regularization()
}

main()
