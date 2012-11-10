library(stats)
library(BGSIMD)

# FUNCTION REMINDER
#
# rdirichlet(N, hyp_thetas) return a matrix N*number(hyp_thetas)
# rmultinorm(N, K, thetas)  return a matrix N*number(thetas), do K tests

multinomial_sample<-function(n, distribution) {
    max.col(t(rmultinom(n, 1, exp(distribution))))
}

sum_log_array<-function(a) {
    log(sum(exp(a[a != -Inf])))
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
        words_counts[category_index]<-words_counts[category_index]-Sampler$corpus[document_index]

        posterior_pi<-rep(-Inf,categories)

        for (category_index in 1:categories) {
            # num = C_x + \gamma_{\pi x} - 1
            # den = N + \sum{\gamma_{\pi i}} - 1
            num<-sum(words_counts[category_index, ])+Sampler$hyp_pi[category_index]-1.

            if (num != 0.) {
                den<-sum(words_counts)+sum(Sampler$hyp_pi)-1.

                label_factor<-num/den
                word_factor<-sum(Sampler$thetas[category_index, ]*words_counts[category_index, ])
                posterior_pi[category_index]<-log(label_factor)+word_factor
            }
        }

        posterior_pi<-posterior_pi-sum_log_array(posterior_pi)
        new_category<-multinomial_sample(1, posterior_pi)
        Sampler$labels[document_index]<-new_category
        words_counts[new_category]<-words_counts[new_category]+Sampler$corpus[document_index,]
        category_counts[new_category]<-category_counts[new_category]+1
    }

    t<-words_counts+Sampler$hyp_thetas

    for (category_index in 1:categories) {
        Sampler$thetas[category_index]<-log(rdirichlet(1, t[category_index]))
    }

    return( list(thetas=Sampler$thetas, labels=Sampler$labels) )
}

estimate_labels<-function(Sampler, iterations, burn_in, lag) {
    estimated_labels<-rep(.0, Sampler$categories)

    # initialize the sampler
    package<-initialize_gibbs_sampler(Sampler)

    Sampler$thetas<-package$thetas
    Sampler$labels<-package$labels

    lag_counter<-lag
    iteration<-1

    estimates<-rep(0.,Sampler$documents);

    while (iteration<iterations) {
        package<-iterate_gibbs_sampler(Sampler)

        Sampler$thetas<-package$thetas
        Sampler$labels<-package$labels

        if (burn_in > 0) {
            burn_in<-burn_in - 1
        } else {
            if (lag_counter > 0) {
                lag_counter<-lag_counter-1
            } else {
                lag_counter<-lag
                print("Sampler$thetas");    print(Sampler$thetas);
                print("Sampler$labels");    print(Sampler$labels);

                estimates<-estimate+Sampler$labels
                iteration<-iteration+1
            }
        }
    }

    estimates/iterations
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

    estimates<-estimate_labels(Sampler, 2, 5, 2)
    print(estimates)
}

main()
