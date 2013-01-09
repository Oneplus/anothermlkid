site<-"http://cran.r-project.org"
if (!require("BGSIMD")) { install.packages("BGSIMD", repos=site) }
if (!require("tm"))     { install.packages("tm", repos=site) }

library(BGSIMD) # for rdirichlet(C, hyp_thetas)

# Sample from multinormial distribution governed by pi
#
#   @param N - size of sample
#   @param distribution - log of multinormial distribution's parameter
#   @return
#       argmax{p(i)} where p(i) is under the multinormial distribution
#       govern by $distribution, the most probable index
multinomial.sample<-function(N, distribution) {
    max.col(t(rmultinom(N, 1, exp(distribution))))
}

# calculate log(exp{a1}+exp{a2}+...+exp{an})
sum.log.array<-function(a) {
    log(sum(exp(a[a > -Inf]-max(a[a>-Inf]))))+max(a[a>-Inf])
}

# Initialize Gibbs Sampler, this process includes
#   - generating a vector of phi \in R^{C}
#   - 
initialize.Gibbs.sampling<-function(C, V, D, hyper_phi, hyper_thetas) {
    # phi \in R^{C}
    # phi[i] is the probability a document happen to be C_i
    phi<-log(rdirchlet(1., hyper_phi))

    # thetas \in R^{C \times V}
    # thetas[i][j] is log of probability word[j] in class[i]
    thetas<-matrix(0., nrow=C, ncol=V)

    for (category_index in 1:C) {
        thetas[category_index, ] <- log(rdirichlet(1, hyper_thetas[category_index,]))
    }

    labels<-multinormial.sample(D, phi)

    return( list( Thetas=thetas, Labels=labels) )
}

iterate.Gibbs.sampling<-function(C, V, D, documents, labels, thetas) {
    # category_counts \in R^{C}
    # category_counts[i] is number of documents with ith category
    category_counts <- rep(0., C)

    # word_counts \in R^{C \times V}
    # word_counts[i][j] is the number of jth word in ith category
    word_counts <- matrix(0., nrow=C, ncol=V)

    for (category_index in 1:C) {
        # count the number of documents in ith category
        category_counts[category_index] <- sum(labels == category_index)
        # count the number of jth word in ith category
        word_counts[category_index, ] <- 
            colSums(documents[labels == category_index, , drop=FALSE])
    }

    # States is represent as <l_1,l_2,...,l_D,phi,thetas;hyper_phi,hyper_thetas>

    for (document_index in 1:D) {
        category <- labels[document_index]

        # erase this document
        word_counts[category, ] <- 
            word_counts[category, ] - documents[document_index, ]
        category_counts[category] <- category_counts[category] - 1

        posterior_phi <- rep(1., C)

        for (category_index in 1:C) {

            # log \Mul_{i=1}^{V} p_i^{\gamma_\theta i}
            num <- sum(word_counts[category_index, ]) + hyp_phi[category_index] - 1.

            if (num != 0) {
                den <- sum(word_counts) + sum(hyp_phi) - 1.
                label_factor <- num / den
                # log \Mul_{i=1}^{V} p_i
                word_factor <- 
                    sum(thetas[category_index, ] * word_counts[category_index, ])

                posterior_phi[category_index] <- log(label_factor) + word_factor
            }
        }

        posterior_phi <- posterior_phi - sum.log.array(posterior_phi)
    }
}


# Function for generating documents
#
#   @param C - size of categories
#   @param V - size of vocabularies
#   @param D - size of documents
#   @param N - number of words in each documents
#   @return 
#       documents - data.frame
#
# HACKED:
#   - $pi is generate from a dirichlet distribution governed
#     by hyp_phi
#   - $thetas is generate from a dirichlet distribution governed
#     by hyp_thetas
generate.documents<-function(C, V, D, N, hyp_phi, hyp_thetas) {
    # generate pi
    phi<-log(rdirichlet(1., hyp_phi))
    # generate thetas
    thetas<-rdirichlet(C, hyp_thetas)

    print(thetas)
    # an empty matrix
    documents<-data.frame(matrix(numeric(0), ncol=(V + 1), nrow=0))

    for (document_index in 1:D) {
        category<-multinomial.sample(1, phi)
        bow<-rmultinom(1, N, thetas[category, ])
        # print(t(bow))
        documents<-rbind(documents, data.frame(label=category, word=t(bow)))
    }

    return(documents)
}

main<-function() {
    C<-3    # size of categories
    V<-5    # size of vocabularies
    D<-20   # number of documents
    N<-100  # number of words each docuemnt

    hyp_phi<-rep(1., C)
    hyp_thetas<-rep(1., V)

    documents<-generate.documents(C, V, D, N, hyp_phi, hyp_thetas)
}

main()
