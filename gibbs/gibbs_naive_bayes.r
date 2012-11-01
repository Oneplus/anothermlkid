library(stats)
library(BGSIMD)

_initalize_gibbs_sampler<-function(GibbsSampler) {
    pi<-log(dirichlet(1, hyp_pi))
    categories<-Sampler$_categories()
    documents<-Sampler$_documents()

    for (category_index in 1:categories) {
        thetas[category_index]<-log(dirichlet(1, hyp_thetas[category_index],))
    }

    Sampler$labels<-sapply(1:documents, multinomial_sample(pi))
}

_iterate_gibbs_sampler<-function(GibbsSampler) {
    documents <- GibbsSampler$_documents()
    vocabulary <- GibbsSampler$_vocabulary()
    categories <- GibbsSampler$_categories()

    category_counts <- rep(numric(0), categories)
    words_counts <- matrix(numric(0), nrow=categories, ncol=vocabulary)

    for (category_index in 1:categories) {
        category_counts[category_index]<-count_nonzero()
    }

    for (document_index in 1:documents) {
    }
}

estimate_labels<-function(GibbsSampler, iterations, burn_in, lag) {
    estimated_labels<-rep(.0, Sampler$_categories)

    _initialize_gibbs_sampler(Sampler)

    lag_counter<-lag

    iteration<-1
    while (iteration<iterations) {
        _iterate_gibbs_sampler()

        if (burn_in > 0) {
            burn_in<-burn_in - 1
        } else {
            if (lag_counter > 0) {
                lag_counter<-lag_counter-1
            } else {
                lag_counter<-lag
                iteration<-iteration+1
            }
        }
    }
}

multinomial_sample<-function(1, distribution) {
    which.max(rmultinom(1, 1, exp(distribution)))
}

generate_corpus<-function(categories, vocabulary, documents) {
    hyp_pi<-rep(1, categories)
    hyp_thetas<-rep(1, vocabulary)

    pi<-log(rdirichlet(1, hyp_pi))
    thetas<-rdirichlet(categories, hyp_thetas)

    labels<-numeric(0)
    corpus<-matrix(numeric(0), ncol=vocabulary, nrow=0)

    for (document_index in 1:documents) {
        category<-multinomial_sample(pi)
        labels<-rbind(labels, category)
        corpus<-rbind(corpus, t(rmultinom(1, vocabulary*100, thetas[category, ])))
    }

    return( list(thetas=log(thetas), corpus=corpus, labels=labels) )
}

main<-function() {
    categories<-10
    vocabulary<-5
    documents<-10

    hyp_pi<-rep(1.,categories)
    hyp_thetas<-matrix(1.,nrow=categories,ncol=vocabulary)

    corpus<-generate_corpus(categories, vocabulary, documents)

    print(corpus$corpus)
}

main()
