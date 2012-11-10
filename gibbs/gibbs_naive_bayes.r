library(stats)
library(BGSIMD)

# FUNCTION REMINDER
#
# rdirichlet(N, hyp_thetas) return a matrix N*number(hyp_thetas)
# rmultinorm(N, K, thetas)  return a matrix N*number(thetas), do K tests

multinomial_sample<-function(distribution) {
    which.max(rmultinom(1, 1, exp(distribution)))
}

initialize_gibbs_sampler<-function(Sampler) {
    # sample parameter for document label multinorm distribution
    pi<-log(rdirichlet(1, Sampler$hyp_pi))

    print("pi"); print(pi)
    categories<-Sampler$categories
    documents<-Sampler$documents

    Sampler$thetas<-matrix(rep(0., length(Sampler$hyp_thetas)), nrow=Sampler$categories)
    print("thetas"); print(Sampler$thetas)
    # for each category in categories, sample
    for (category_index in 1:categories) {
        Sampler$thetas[category_index, ]<-log(rdirichlet(1, Sampler$hyp_thetas[category_index, ]))
    }

    Sampler$labels<-sapply(1:documents, multinomial_sample(pi))
}

iterate_gibbs_sampler<-function(Sampler) {
    documents <- Sampler$documents
    vocabulary <- Sampler$vocabulary
    categories <- Sampler$categories

    category_counts <- rep(numric(0), categories)
    words_counts <- matrix(numric(0), nrow=categories, ncol=vocabulary)

    for (category_index in 1:categories) {
        category_counts[category_index]<-count_nonzero()
    }

    for (document_index in 1:documents) {
    }
}

estimate_labels<-function(Sampler, iterations, burn_in, lag) {
    estimated_labels<-rep(.0, Sampler$categories)

    # initialize the sampler
    initialize_gibbs_sampler(Sampler)

    lag_counter<-lag
    iteration<-1
    while (iteration<iterations) {
        iterate_gibbs_sampler()

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
    categories<-3
    vocabulary<-5
    documents<-7

    hyp_pi<-rep(1.,categories)
    hyp_thetas<-matrix(1.,nrow=categories,ncol=vocabulary)

    package<-generate_corpus(categories, vocabulary, documents)

    #print(exp(package$thetas))
    #print(package$corpus)
    #print(package$labels)

    hyp_pi<-rep(1, categories)
    hyp_thetas<-matrix(rep(1, categories*vocabulary), nrow=categories)

    Sampler<-list(categories=categories,
                vocabulary=vocabulary,
                documents=documents,
                corpus=package$corpus,
                hyp_pi=hyp_pi,
                hyp_thetas=hyp_thetas)

    estimate_labels(Sampler, 100, 5, 2)
}

main()
