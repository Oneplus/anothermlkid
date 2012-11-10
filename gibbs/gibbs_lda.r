library(stats)
library(BGSIMD)

# FUNCTION REMINDER
#
# rdirichlet(N, hyp_thetas) return a matrix N*number(hyp_thetas)
# rmultinorm(N, K, thetas)  return a matrix N*number(thetas), do K tests

multinomial_sample<-function(n, distribution) {
    max.col(t(rmultinom(n, 1, exp(distribution))))
}

initialize_gibbs_sampler<-function(Sampler) {
    # sample parameter for document label multinorm distribution
    pi<-log(rdirichlet(1, Sampler$hyp_pi))

    print("pi"); print(pi)
    categories<-Sampler$categories
    documents<-Sampler$documents

    thetas<-matrix(rep(0., length(Sampler$hyp_thetas)), nrow=categories)
    print("thetas"); print(thetas)
    # for each category in categories, sample
    for (category_index in 1:categories) {
        thetas[category_index, ]<-log(rdirichlet(1, Sampler$hyp_thetas[category_index, ]))
    }

    labels<-multinomial_sample(documents, pi)

    return( list(thetas=thetas, labels=labels) )
}

iterate_gibbs_sampler<-function(Sampler) {
    documents <- Sampler$documents
    vocabulary <- Sampler$vocabulary
    categories <- Sampler$categories

    category_counts <- rep(0., categories)
    words_counts <- matrix(rep(0, length(Sampler$hyp_thetas)), nrow=categories)

    for (category_index in 1:categories) {
        category_counts[category_index]<-sum(category_index == Sampler$labels)
        words_counts[category_index,]<-colSums(Sampler$corpus[category_index == Sampler$labels,,drop=FALSE])
    }

    for (document_index in 1:documents) {
        category_index<-Sampler$labels[document_index]
        word_counts[category_index]<-word_counts[category_index]-Sampler$corpus[document_index]

        posterior_pi<-rep(0.,categories)

        for (category_index in 1:categories) {

        }
    }
}

estimate_labels<-function(Sampler, iterations, burn_in, lag) {
    estimated_labels<-rep(.0, Sampler$categories)

    # initialize the sampler
    package<-initialize_gibbs_sampler(Sampler)

    Sampler$thetas<-package$thetas
    Sampler$labels<-package$labels

    print("Sampler$thetas");    print(Sampler$thetas);
    print("Sampler$labels");    print(Sampler$labels);

    lag_counter<-lag
    iteration<-1

    while (iteration<iterations) {
        iterate_gibbs_sampler(Sampler)

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
        category<-multinomial_sample(1, pi)
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
    print(package$corpus)
    #print(package$labels)

    hyp_pi<-rep(1, categories)
    hyp_thetas<-matrix(rep(1, categories*vocabulary), nrow=categories)

    Sampler<-list(categories=categories,
                vocabulary=vocabulary,
                documents=documents,
                corpus=package$corpus,
                hyp_pi=hyp_pi,
                hyp_thetas=hyp_thetas)

    estimate_labels(Sampler, 2, 5, 2)
}

main()
