
naive_knn<-function( K ) {
    raw<-read.table("data/semeion.dat", header=FALSE)
    raw<-data.matrix(raw)

    N<-nrow(raw)
    train_size<-floor(N*.7)

    train_set<-sample(1:N,train_size,replace=FALSE)
    test_set<-setdiff(1:N,train_set)

    train_x<-raw[train_set,1:256]
    train_y<-raw[train_set,257:266]

    test_x<-raw[test_set,1:256]
    test_y<-raw[test_set,257:266]
    test_y<-sapply(1:nrow(test_x), function(j) { order(test_y[j,], decreasing=T)[1] })

    pred_y<-sapply( 1:nrow(test_x),
        function(i) {
            order( colSums( train_y[ order(
                sapply( 1:nrow(train_x), function(j) {sum((train_x[j,]-test_x[i,])^2)} )
                        )[1:K], ] ), decreasing=T )[1]
        } )

    # print(test_y)
    # print(pred_y)
    cat("accuracy: ", sum(ifelse(pred_y-test_y==0,1,0))/nrow(test_x)*100.)
}


main<-function() {
    naive_knn( 3 )
}
main()
