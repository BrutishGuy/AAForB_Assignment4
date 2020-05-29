# read in some stopwords:
library(tm)
library(sentimentr)
library(readr)
library(readxl)
library(lda)
library(LDAvis)

df_posts <- read_csv("../data/poststhreemainsubreddits_uncompleteshortlist_of_users.csv")

stop_words <- stopwords("SMART")

# pre-processing:
df_posts$title <- gsub("'", "", df_posts$title)  # remove apostrophes
df_posts$title <- gsub("[[:punct:]]", " ", df_posts$title)  # replace punctuation with space
df_posts$title <- gsub("[[:cntrl:]]", " ", df_posts$title)  # replace control characters with space
df_posts$title <- gsub("^[[:space:]]+", "", df_posts$title) # remove whitespace at beginning of documents
df_posts$title <- gsub("[[:space:]]+$", "", df_posts$title) # remove whitespace at end of documents
df_posts$title <- tolower(df_posts$title)  # force to lowercase


df_sentences <- get_sentences(df_posts$title)
df_posts_sentiment <- sentiment_by(df_sentences, averaging.function = sentimentr::average_weighted_mixed_sentiment)

df_posts$avg_sentiment <- df_posts_sentiment$ave_sentiment
df_posts$sd_sentiment <- df_posts_sentiment$sd_sentiment


# tokenize on space and output as a list:
doc.list <- strsplit(df_posts$title, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  

# MCMC and model tuning parameters:
K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:


set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 24 minutes on laptop

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

LDAResults <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)


# create the JSON object to feed the visualization:
json <- createJSON(phi = LDAResults$phi, 
                   theta = LDAResults$theta, 
                   doc.length = LDAResults$doc.length, 
                   vocab = LDAResults$vocab, 
                   term.frequency = LDAResults$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)



# Further testing
set.seed(357)
t1 <- Sys.time()
fit <- slda.em(documents = documents, K = K, vocab = vocab, 
                                   num.e.iterations = G, alpha = alpha, 
                                   eta = eta)
t2 <- Sys.time()
t2 - t1  # about 24 minutes on laptop

for (i in 0:length(documents)) {
  if (length(documents[i]) == 0) {
    documents <- documents[-i]
  }
}
  

