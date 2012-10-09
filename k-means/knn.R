gauss_2d_sample <- function(N) {
    c(c(rnorm(N/2, -1.5, 1), rnorm(N/2, 1.5, 1)), c(rnorm(N/2,-1.5,1), rnorm(N/2, 1.5,1)))
}

main <- function() {
    N = 100
    x <- gauss_2d_sample(N)
    png("test.png")
    plot(x)
}

main()
